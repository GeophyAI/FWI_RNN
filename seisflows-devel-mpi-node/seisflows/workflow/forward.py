#!/usr/bin/env python3
"""
The simplest simulation workflow you can run is a large number of forward
simulations to generate synthetics from a velocity model. Therefore the
Forward class represents the BASE workflow. All other workflows will build off
of the scaffolding defined by the Forward class.
"""

import os, shutil
from time import asctime
from glob import glob

from seisflows import logger
from seisflows.tools import msg, unix
from seisflows.tools.config import Dict
from seisflows.tools.model import Model


class Forward:
    """
    Forward Workflow
    ----------------
    Defines foundational structure for Workflow module. When used standalone 
    is in charge of running forward solver in parallel and (optionally) 
    calculating data-synthetic misfit and adjoint sources.

    Parameters
    ----------
    :type modules: list of module
    :param modules: instantiated SeisFlows modules which should have been
        generated by the function `seisflows.config.import_seisflows` with a
        parameter file generated by seisflows.configure
    :type data_case: str
    :param data_case: How to address 'data' in the workflow, available options:
        'data': real data will be provided by the user in
        `path_data/{source_name}` in the same format that the solver will
        produce synthetics (controlled by `solver.format`) OR
        synthetic': 'data' will be generated as synthetic seismograms using
        a target model provided in `path_model_true`. If None, workflow will
        not attempt to generate data.
    :type stop_after: str
    :param stop_after: optional name of task in task list (use
        `seisflows print tasks` to get task list for given workflow) to stop
        workflow after, allowing user to prematurely stop a workflow to explore
        intermediate results or debug.
    :type export_traces: bool
    :param export_traces: export all waveforms that are generated by the
        external solver to `path_output`. If False, solver traces stored in
        scratch may be discarded at any time in the workflow
    :type export_residuals: bool
    :param export_residuals: export all residuals (data-synthetic misfit) that
        are generated by the external solver to `path_output`. If False,
        residuals stored in scratch may be discarded at any time in the 
        workflow

    Paths
    -----
    :type workdir: str
    :param workdir: working directory in which to perform a SeisFlows workflow.
        SeisFlows internal directory structure will be created here. Default cwd
    :type path_output: str
    :param path_output: path to directory used for permanent storage on disk.
        Results and exported scratch files are saved here.
    :type path_data: str
    :param path_data: path to any externally stored data required by the solver
    :type path_state_file: str
    :param path_state_file: path to a text file used to track the current
        status of a workflow (i.e., what functions have already been completed),
        used for checkpointing and resuming workflows
    :type path_model_init: str
    :param path_model_init: path to the starting model used to calculate the
        initial misfit. Must match the expected `solver_io` format.
    :type path_model_true: str
    :param path_model_true: path to a target model if `case`=='synthetic' and
        a set of synthetic 'observations' are required for workflow.
    :type path_eval_grad: str
    :param path_eval_grad: scratch path to store files for gradient evaluation,
        including models, kernels, gradient and residuals.
    ***
    """
    def __init__(self, modules=None, data_case="data", stop_after=None,
                 export_traces=False, export_residuals=False,
                 workdir=os.getcwd(), path_output=None, path_data=None,
                 path_state_file=None, path_model_init=None,
                 path_model_true=None, path_eval_grad=None, **kwargs):
        """
        Set default forward workflow parameters

        :type modules: list
        :param modules: list of sub-modules that will be established as class
            attributes by the setup() function. Should not need to be set by the
            user
        """
        # Keep modules hidden so that seisflows configure doesnt count them
        # as 'parameters'
        self._modules = modules

        self.data_case = data_case
        self.stop_after = stop_after
        self.export_traces = export_traces
        self.export_residuals = export_residuals

        self.path = Dict(
            workdir=workdir,
            scratch=os.path.join(workdir, "scratch"),
            eval_grad=path_eval_grad or
                      os.path.join(workdir, "scratch", "eval_grad"),
            output=path_output or os.path.join(workdir, "output"),
            model_init=path_model_init,
            model_true=path_model_true,
            state_file=path_state_file or
                       os.path.join(workdir, "sfstate.txt"),
            data=path_data,
        )

        self._required_modules = ["system", "solver"]
        self._acceptable_data_cases = ["data", "synthetic"]
        self._optional_modules = ["preprocess"]

        # Read in any existing state file which keeps track of workflow tasks
        self._states = {}
        if os.path.exists(self.path.state_file):
            for line in open(self.path.state_file, "r").readlines():
                if line.startswith("#"):
                    continue
                key, val = line.strip().split(":")
                self._states[key] = val.strip()

    @property
    def task_list(self):
        """
        USER-DEFINED TASK LIST. This property defines a list of class methods
        that take NO INPUT and have NO RETURN STATEMENTS. This defines your
        linear workflow, i.e., these tasks are to be run in order from start to
        finish to complete a workflow.

        This excludes 'check' (which is run during 'import_seisflows') and
        'setup' which should be run separately

        .. note::
            For workflows that require an iterative approach (e.g. inversion),
            this task list will be looped over, so ensure that any setup and
            teardown tasks (run once per workflow, not once per iteration) are
            not included.

        :rtype: list
        :return: list of methods to call in order during a workflow
        """
        return [self.evaluate_initial_misfit]

    def check(self):
        """
        Check that workflow has required modules. Run their respective checks
        """
        # Check that required modules have been instantiated
        for req_mod in self._required_modules:
            assert(self._modules[req_mod]), (
                f"'{req_mod}' is a required module for workflow " 
                f"'{self.__class__.__name__}'"
            )
            # Make sure that the modules are actually instances (not e.g., str)
            assert(hasattr(self._modules[req_mod], "__class__")), \
                f"workflow attribute {req_mod} must be an instance"

            # Run check function of these modules
            self._modules[req_mod].check()

        # Tell the user whether optional modules are instantiated
        for opt_mod in self._optional_modules:
            if self._modules[opt_mod]:
                self._modules[opt_mod].check()
            else:
                logger.warning(f"optional module '{opt_mod}' has not been "
                               f"instantiated, some functionality of the "
                               f"'{self.__class__.__name__}' workflow may be "
                               f"skipped")

        # If we are using the preprocessing module, we must have either
        # 1) real data located in `path.data`, or 2) a target model to generate
        # synthetic data, locaed in `path.model_true`
        if self.data_case is not None and self._modules.preprocess:
            assert(self.data_case.lower() in self._acceptable_data_cases), \
                f"`data_case` must be in {self._acceptable_data_cases}"
            if self.data_case.lower() == "data":
                assert(self.path.data is not None and
                       os.path.exists(self.path.data)), \
                    f"importing data with `data_case`=='data' requires " \
                    f"'path_data' to exist"
            elif self.data_case.lower() == "synthetic":
                assert(self.path.model_true is not None and
                       os.path.exists(self.path.model_true)), \
                    f"creating data with `data_case`=='synthetic' requires " \
                    f"'path_model_true' to exist and point to a target model"
        else:
            logger.warning(f"`workflow.data_case` is None, SeisFlows will not "
                           f"be able to find data for data-synthetic comparison"
                           )

        if self.stop_after is not None:
            _task_names = [task.__name__ for task in self.task_list]
            assert(self.stop_after in _task_names), \
                f"workflow parameter `stop_after` must match {_task_names}"

    def setup(self):
        """
        Assigns modules as attributes of the workflow. I.e., `self.solver` to
        access the solver module (or `workflow.solver` from outside class)

        Makes required path structure for the workflow, runs setup functions
        for all the required modules of this workflow.
        """
        logger.info(msg.mjr(f"SETTING UP {self.__class__.__name__.upper()} "
                            f"WORKFLOW"))

        # Create the desired directory structure
        for path in self.path.values():
            if path is not None and not os.path.splitext(path)[-1]:
                unix.mkdir(path)

        # Run setup() for each of the required modules
        for req_mod in self._required_modules:
            logger.debug(
                f"running setup for module "
                f"'{req_mod}.{self._modules[req_mod].__class__.__name__}'"
            )
            self._modules[req_mod].setup()

        # Run setup() for each of the instantiated modules
        for opt_mod in self._optional_modules:
            if self._modules[opt_mod] and opt_mod not in self._required_modules:
                logger.debug(
                    f"running setup for module "
                    f"'{opt_mod}.{self._modules[opt_mod].__class__.__name__}'"
                )
                self._modules[opt_mod].setup()

        # Generate the state file to keep track of task completion
        if not os.path.exists(self.path.state_file):
            with open(self.path.state_file, "w") as f:
                f.write(f"# SeisFlows State File\n")
                f.write(f"# {asctime()}\n")
                f.write(f"# Acceptable states: 'completed', 'failed', "
                        f"'pending'\n")
                f.write(f"# =======================================\n")

        # Distribute modules to the class namespace. We don't do this at init
        # incase _modules was set as NoneType
        self.solver = self._modules.solver  # NOQA
        self.system = self._modules.system  # NOQA
        self.preprocess = self._modules.preprocess  # NOQA

    def checkpoint(self):
        """
        Saves active SeisFlows working state to disk as a text files such that
        the workflow can be resumed following a crash, pause or termination of
        workflow.
        """
        # Grab State file header values
        with open(self.path.state_file, "r") as f:
            lines = f.readlines()

        with open(self.path.state_file, "w") as f:
            # Rewrite header values
            for line in lines:
                if line.startswith("#"):
                    f.write(line)
            for key, val in self._states.items():
                f.write(f"{key}: {val}\n")

    def run(self):
        """
        Call the Task List in order to 'run' the workflow. Contains logic for
        to keep track of completed tasks and avoids re-running tasks that have
        previously been completed (e.g., if you are restarting your workflow)
        """
        logger.info(msg.mjr(f"RUNNING {self.__class__.__name__.upper()} "
                            f"WORKFLOW"))

        for func in self.task_list:
            # Skip over functions which have already been completed
            if (func.__name__ in self._states.keys()) and (
                    self._states[func.__name__] == "completed"):
                logger.info(f"'{func.__name__}' has already been run, skipping")
                continue
            # Otherwise attempt to run functions that have failed or are
            # encountered for the first time
            else:
                try:
                    func()
                    self._states[func.__name__] = "completed"
                    self.checkpoint()
                except Exception as e:
                    self._states[func.__name__] = "failed"
                    self.checkpoint()
                    raise
            # Allow user to prematurely stop a workflow after a given task
            if self.stop_after and func.__name__ == self.stop_after:
                logger.info(f"stop workflow at `stop_after`: {self.stop_after}")
                break

        self.checkpoint()

    def evaluate_initial_misfit(self):
        """
        Evaluate the initial model misfit. This requires setting up 'data'
        before generating synthetics, which is either copied from user-supplied
        directory or running forward simulations with a target model. Forward
        simulations are then run and prepocessing compares data-synthetic misfit

        .. note::
            This is run altogether on system to save on queue time waits,
            because we are potentially running two simulations back to back.
        """
        logger.info(msg.mnr("EVALUATING MISFIT FOR INITIAL MODEL"))

        # Load in the initial model and check its poissons ratio
        if self.path.model_init:
            logger.info("checking initial model parameters")
            _model = Model(os.path.join(self.path.model_init))
            _model.check()
        if self.path.model_true:
            logger.info("checking true/target model parameters")
            _model = Model(os.path.join(self.path.model_true))
            _model.check()

        # If no preprocessing module, than all of the additional functions for
        # working with `data` are unncessary.
        if self.preprocess:
            run_list = [self.prepare_data_for_solver,
                        self.run_forward_simulations,
                        self.evaluate_objective_function]
        else:
            run_list = [self.run_forward_simulations]

        self.system.run(run_list, path_model=self.path.model_init,
                        save_residuals=os.path.join(self.path.eval_grad,
                                                    "residuals.txt")
                        )

    def prepare_data_for_solver(self, **kwargs):
        """
        Determines how to provide data to each of the solvers. Either by copying
        data in from a user-provided path, or generating synthetic 'data' using
        a target model.

        .. note ::
            Must be run by system.run() so that solvers are assigned individual
            task ids and working directories
        """
        #logger.info(f"preparing observation data for source "
        #            f"{self.solver.source_name}")
        
        if self.data_case == "data":

            if int(self.solver.source_name)==1:
                logger.info(f"preparing observation data for sources")
                for source_name in self.solver.source_names:
                    src = os.path.join(self.path.data, source_name, "*")
                    dst = os.path.join(self.solver.path.solver, source_name, "traces", "obs", "")
                    #logger.info(f"copying data from {src} to {dst}")
                    if glob(os.path.join(dst, "*")):
                        continue
                    os.system('cp {} {}'.format(src, dst))

        elif self.data_case == "synthetic":
            # Figure out where to export waveform files to, if requested
            if self.export_traces:
                export_traces = os.path.join(self.path.output,
                                             self.solver.source_name, "obs")
            else:
                export_traces = False

            # Run the forward solver with target model and save traces the 'obs'
            logger.info(f"running forward simulation w/ target model for "
                        f"{self.solver.source_name}")
            self.solver.import_model(path_model=self.path.model_true)
            self.solver.forward_simulation(
                save_traces=os.path.join(self.solver.cwd, "traces", "obs"),
                export_traces=export_traces
            )

    def run_forward_simulations(self, path_model, no_backward=False, **kwargs):
        """
        Performs forward simulation for a single given event.

        .. note::
            if PAR.PREPROCESS == None, will not perform misfit quantification

        .. note::
            Must be run by system.run() so that solvers are assigned individual
            task ids/ working directories.
        """
        assert(os.path.exists(path_model)), \
            f"Model path for objective function does not exist"

        #logger.info(f"evaluating objective function for source "
        #            f"{self.solver.source_name}")

        # Calculate the syn data only once
        if int(self.solver.source_name)>1:
            #logger.debug(f"source: {self.solver.source_name}>1, skipping forward_simulation")
            return

        logger.debug(f"running forward simulation with "
                     f"'{self.solver.__class__.__name__}'")

        # Figure out where to export waveform files to, if requested
        # path will look like: 'output/solver/001/syn/NN.SSS.BXY.semd'
        if self.export_traces:
            export_traces = os.path.join(self.path.output, "solver",
                                         self.solver.source_name, "syn")
        else:
            export_traces = False

        # Run the forward simulation with the given input model
        self.solver.import_model(path_model=path_model)
        self.solver.forward_simulation(
            save_traces=os.path.join(self.solver.cwd, "traces", "syn"),
            export_traces=export_traces, no_backward=no_backward,
        )

    def evaluate_objective_function(self, save_residuals=False, **kwargs):
        """
        Uses the preprocess module to evaluate the misfit/objective function
        given synthetics generated during forward simulations

        .. note::
            Must be run by system.run() so that solvers are assigned individual
            task ids/ working directories.
        """
        if self.preprocess is None:
            logger.debug("no preprocessing module selected, will not evaluate "
                         "objective function")
            return

        #logger.debug(f"quantifying misfit with "
        #             f"'{self.preprocess.__class__.__name__}' "
        #             f"for {self.solver.source_name}")
        self.preprocess.quantify_misfit(
            source_name=self.solver.source_name,
            source_names=self.solver.source_names,
            save_adjsrcs=os.path.join(self.solver.cwd, "traces", "adj"),
            save_residuals=save_residuals,
            nproc=self.solver.nproc,
        )
