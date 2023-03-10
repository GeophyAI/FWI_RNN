#!/usr/bin/env python3
"""
The SeisFlows Preprocessing module is in charge of interacting with seismic
data (observed and synthetic). It should contain functionality to read and write
seismic data, apply preprocessing such as filtering, quantify misfit,
and write adjoint sources that are expected by the solver.
"""
import os
import numpy as np
from glob import glob
from obspy import read as obspy_read
from obspy import Stream, Trace, UTCDateTime
from joblib import Parallel, delayed

from seisflows import logger
from seisflows.tools import signal, unix
from seisflows.tools.config import Dict, get_task_id

from seisflows.plugins.preprocess import misfit as misfit_functions
from seisflows.plugins.preprocess import adjoint as adjoint_sources


class Default:
    """
    Default Preprocess
    ------------------
    Data processing for seismic traces, with options for data misfit,
    filtering, normalization and muting.

    Parameters
    ----------
    :type data_format: str
    :param data_format: data format for reading traces into memory. For
        available see: seisflows.plugins.preprocess.readers
    :type misfit: str
    :param misfit: misfit function for waveform comparisons. For available
        see seisflows.plugins.preprocess.misfit
    :type backproject: str
    :param backproject: backprojection function for migration, or the
        objective function in FWI. For available see
        seisflows.plugins.preprocess.adjoint
    :type normalize: str
    :param normalize: Data normalization parameters used to normalize the
        amplitudes of waveforms. Choose from two sets:
        ENORML1: normalize per event by L1 of traces; OR
        ENORML2: normalize per event by L2 of traces;
        &
        TNORML1: normalize per trace by L1 of itself; OR
        TNORML2: normalize per trace by L2 of itself
    :type filter: str
    :param filter: Data filtering type, available options are:
        BANDPASS (req. MIN/MAX PERIOD/FREQ);
        LOWPASS (req. MAX_FREQ or MIN_PERIOD);
        HIGHPASS (req. MIN_FREQ or MAX_PERIOD)
    :type min_period: float
    :param min_period: Minimum filter period applied to time series.
        See also MIN_FREQ, MAX_FREQ, if User defines FREQ parameters, they
        will overwrite PERIOD parameters.
    :type max_period: float
    :param max_period: Maximum filter period applied to time series. See
        also MIN_FREQ, MAX_FREQ, if User defines FREQ parameters, they will
        overwrite PERIOD parameters.
    :type min_freq: float
    :param min_freq: Maximum filter frequency applied to time series,
        See also MIN_PERIOD, MAX_PERIOD, if User defines FREQ parameters,
        they will overwrite PERIOD parameters.
    :type max_freq: float
    :param max_freq: Maximum filter frequency applied to time series,
        See also MIN_PERIOD, MAX_PERIOD, if User defines FREQ parameters,
        they will overwrite PERIOD parameters.
    :type mute: list
    :param mute: Data mute parameters used to zero out early / late
        arrivals or offsets. Choose any number of:
        EARLY: mute early arrivals;
        LATE: mute late arrivals;
        SHORT: mute short source-receiver distances;
        LONG: mute long source-receiver distances

    Paths
    -----
    :type path_preprocess: str
    :param path_preprocess: scratch path for all preprocessing processes,
        including saving files
    ***
    """
    def __init__(self, data_format="ascii", misfit="waveform",
                 adjoint="waveform", normalize=None, filter=None,
                 min_period=None, max_period=None, min_freq=None, max_freq=None,
                 mute=None, early_slope=None, early_const=None, late_slope=None,
                 late_const=None, short_dist=None, long_dist=None,
                 workdir=os.getcwd(), path_preprocess=None, path_solver=None,
                 **kwargs):
        """
        Preprocessing module parameters

        .. note::
            Paths listed here are shared with `workflow.forward` and so are not
            included in the class docstring.

        :type workdir: str
        :param workdir: working directory in which to look for data and store
        results. Defaults to current working directory
        :type path_preprocess: str
        :param path_preprocess: scratch path for all preprocessing processes,
            including saving files
        """
        self.data_format = data_format.upper()
        self.misfit = misfit
        self.adjoint = adjoint
        self.normalize = normalize

        self.filter = filter
        self.min_period = min_period
        self.max_period = max_period
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.mute = mute or []
        self.normalize = normalize or []

        # Mute arrivals sub-parameters
        self.early_slope = early_slope
        self.early_const = early_const
        self.late_slope = late_slope
        self.late_const = late_const
        self.short_dist = short_dist
        self.long_dist = long_dist

        self.path = Dict(
            scratch=path_preprocess or os.path.join(workdir, "scratch",
                                                    "preprocess"),
            solver=path_solver or os.path.join(workdir, "scratch", "solver")
        )

        self._acceptable_data_formats = ["SU", "ASCII"]

        # Misfits and adjoint sources are defined by the available functions
        # in each of these plugin files. Drop hidden variables from dir()
        self._acceptable_misfits = [_ for _ in dir(misfit_functions)
                                    if not _.startswith("_")]
        self._acceptable_adjsrcs = [_ for _ in dir(adjoint_sources)
                                    if not _.startswith("_")]

        # Internal attributes used to keep track of inversion workflows
        self._iteration = None
        self._step_count = None
        self._source_names = None

    def check(self):
        """ 
        Checks parameters and paths
        """
        if self.misfit:
            assert(self.misfit in self._acceptable_misfits), \
                f"preprocess.misfit must be in {self._acceptable_misfits}"
        if self.adjoint:
            assert(self.adjoint in self._acceptable_adjsrcs), \
                f"preprocess.misfit must be in {self._acceptable_adjsrcs}"

        # Data normalization option
        if self.normalize:
            acceptable_norms = {"TNORML1", "TNORML2", "ENORML1", "ENORML2"}
            chosen_norms = [_.upper() for _ in self.normalize]
            assert(set(chosen_norms).issubset(acceptable_norms))

        # Data muting options
        if self.mute:
            acceptable_mutes = {"EARLY", "LATE", "LONG", "SHORT"}
            chosen_mutes = [_.upper() for _ in self.mute]
            assert(set(chosen_mutes).issubset(acceptable_mutes))
            if "EARLY" in chosen_mutes:
                assert(self.early_slope is not None)
                assert(self.early_slope >= 0.)
                assert(self.early_const is not None)
            if "LATE" in chosen_mutes:
                assert(self.late_slope is not None)
                assert(self.late_slope >= 0.)
                assert(self.late_const is not None)
            if "SHORT" in chosen_mutes:
                assert(self.short_dist is not None)
                assert (self.short_dist > 0)
            if "LONG" in chosen_mutes:
                assert(self.long_dist is not None)
                assert (self.long_dist > 0)

        # Data filtering options that will be passed to ObsPy filters
        if self.filter:
            acceptable_filters = ["BANDPASS", "LOWPASS", "HIGHPASS"]
            assert self.filter.upper() in acceptable_filters, \
                f"self.filter must be in {acceptable_filters}"

            # Set the min/max frequencies and periods, frequency takes priority
            if self.min_freq is not None:
                self.max_period = 1 / self.min_freq
            elif self.max_period is not None:
                self.min_freq = 1 / self.max_period

            if self.max_freq is not None:
                self.min_period = 1 / self.max_freq
            elif self.min_period is not None:
                self.max_freq =  1 / self.min_period

            # Check that the correct filter bounds have been set
            if self.filter.upper() == "BANDPASS":
                assert(self.min_freq is not None and
                       self.max_freq is not None), \
                    ("BANDPASS filter PAR.MIN_PERIOD and PAR.MAX_PERIOD or " 
                     "PAR.MIN_FREQ and PAR.MAX_FREQ")
            elif self.filter.upper() == "LOWPASS":
                assert(self.max_freq is not None or
                       self.min_period is not None),\
                    "LOWPASS requires PAR.MAX_FREQ or PAR.MIN_PERIOD"
            elif self.filter.upper() == "HIGHPASS":
                assert(self.min_freq is not None or
                       self.max_period is not None),\
                    "HIGHPASS requires PAR.MIN_FREQ or PAR.MAX_PERIOD"

            # Check that filter bounds make sense, by this point, MIN and MAX
            # FREQ and PERIOD should be set, so we just check the FREQ
            assert(0 < self.min_freq < np.inf), "0 < PAR.MIN_FREQ < inf"
            assert(0 < self.max_freq < np.inf), "0 < PAR.MAX_FREQ < inf"
            assert(self.min_freq < self.max_freq), (
                "PAR.MIN_FREQ < PAR.MAX_FREQ"
            )

        assert(self.data_format.upper() in self._acceptable_data_formats), \
            f"data format must be in {self._acceptable_data_formats}"

    def setup(self):
        """
        Sets up data preprocessing machinery by dynamicalyl loading the
        misfit, adjoint source type, and specifying the expected file type
        for input and output seismic data.
        """
        unix.mkdir(self.path.scratch)

    def read(self, fid):
        """
        Waveform reading functionality. Imports waveforms as Obspy streams

        :type fid: str
        :param fid: path to file to read data from
        :rtype: obspy.core.stream.Stream
        :return: ObsPy stream containing data stored in `fid`
        """
        st = None
        if self.data_format.upper() == "SU":
            st = obspy_read(fid)#obspy_read(fid, format="SU", byteorder="<")
        elif self.data_format.upper() == "ASCII":
            st = read_ascii(fid)
        return st

    def write(self, st, fid):
        """
        Waveform writing functionality. Writes waveforms back to format that
        SPECFEM recognizes

        :type st: obspy.core.stream.Stream
        :param st: stream to write
        :type fid: str
        :param fid: path to file to write stream to
        """
        if self.data_format.upper() == "SU":
            for tr in st:
                # Work around for ObsPy data type conversion
                tr.data = tr.data.astype(np.float32)
            max_delta = 0.065535
            dummy_delta = max_delta

            if st[0].stats.delta > max_delta:
                for tr in st:
                    tr.stats.delta = dummy_delta

            # Write data to file
            st.write(fid, format="SU")

        elif self.data_format.upper() == "ASCII":
            for tr in st:
                # Float provides time difference between starttime and default
                time_offset = float(tr.stats.starttime)
                data_out = np.vstack((tr.times() + time_offset, tr.data)).T
                np.savetxt(fid, data_out, ["%13.7f", "%17.7f"])

    def _calculate_misfit(self, **kwargs):
        """Wrapper for plugins.preprocess.misfit misfit/objective function"""
        if self.misfit is not None:
            return getattr(misfit_functions, self.misfit)(**kwargs)
        else:
            return None

    def _generate_adjsrc(self, **kwargs):
        """Wrapper for plugins.preprocess.adjoint source/backproject function"""
        if self.adjoint is not None:
            return getattr(adjoint_sources, self.adjoint)(**kwargs)
        else:
            return None

    def initialize_adjoint_traces(self, data_filenames, output):
        """
        SPECFEM requires that adjoint traces be present for every matching
        synthetic seismogram. If an adjoint source does not exist, it is
        simply set as zeros. This function creates all adjoint traces as
        zeros, to be filled out later

        Appends '.adj. to the solver filenames as expected by SPECFEM (if they
        don't already have that extension)

        TODO there are some sem2d and 3d specific tasks that are not carried
        TODO over here, were they required?

        :type data_filenames: list of str
        :param data_filenames: existing solver waveforms to read from.
            These will be copied, zerod out, and saved to path `save`. Should
            come from solver.data_filenames
        :type output: str
        :param output: path to save the new adjoint traces to.
        """
        for fid in data_filenames:
            st = self.read(fid=fid).copy()
            fid = os.path.basename(fid)  # drop any path before filename
            for tr in st:
                tr.data *= 0

            adj_fid = self._rename_as_adjoint_source(fid)

            # Write traces back to the adjoint trace directory
            self.write(st=st, fid=os.path.join(output, adj_fid))

    def _rename_as_adjoint_source(self, fid):
        """
        Rename synthetic waveforms into filenames consistent with how SPECFEM
        expects adjoint sources to be named. Usually this just means adding
        a '.adj' to the end of the filename

        TODO how does SPECFEM3D_GLOBE expect this? filenames end with .sem.ascii
            so the .ascii will get replaced. Is that okay?
        """
        if not fid.endswith(".adj"):
            if self.data_format.upper() == "SU":
                fid = f"{fid}.adj"
            elif self.data_format.upper() == "ASCII":
                og_extension = os.path.splitext(fid)[-1]  # e.g., .semd
                fid = fid.replace(og_extension, ".adj")
        return fid

    def _setup_quantify_misfit(self, source_name):
        """
        Gather waveforms from the Solver scratch directory and
        """
        source_name = source_name or self._source_names[get_task_id()]

        obs_path = os.path.join(self.path.solver, source_name, "traces", "obs")
        syn_path = os.path.join(self.path.solver, source_name, "traces", "syn")

        observed = sorted(glob(os.path.join(obs_path, "*")))
        synthetic = sorted(glob(os.path.join(syn_path, "*")))

        assert(len(observed) == len(synthetic)), (
            f"number of observed traces does not match length of synthetic for "
            f"source: {source_name}"
        )

        assert(len(observed) != 0 and len(synthetic) != 0), \
            f"cannot quantify misfit, missing observed or synthetic traces"

        return observed, synthetic

    def _setup_quantify_misfit_line_search(self, source_name):
        """
        Gather waveforms from the Solver scratch directory and
        """
        source_name = source_name or self._source_names[get_task_id()]

        obs_path = os.path.join(self.path.solver, source_name, "traces", "obs")
        syn_path = os.path.join(self.path.solver, source_name, "traces", "syn")
        adj_path = os.path.join(self.path.solver, source_name, "traces", "adj")

        observed = sorted(glob(os.path.join(obs_path, "*")))
        synthetic = sorted(glob(os.path.join(syn_path, "*")))
        adjoint = sorted(glob(os.path.join(adj_path, "*")))

        assert(len(observed) == len(synthetic) == len(adjoint)), (
            f"number of observed traces does not match length of synthetic for "
            f"source: {source_name}"
        )

        assert(len(observed) != 0 and len(synthetic) != 0 and len(adjoint) != 0 ), \
            f"cannot quantify misfit, missing observed or synthetic traces or adjoint traces"

        return observed, synthetic, adjoint

    def quantify_misfit(self, source_name=None, source_names=None, save_residuals=None,
                        save_adjsrcs=None, iteration=1, step_count=0, nproc=1,
                        **kwargs):
        """
        Prepares solver for gradient evaluation by writing residuals and
        adjoint traces. Meant to be called by solver.eval_func().

        Reads in observed and synthetic waveforms, applies optional
        preprocessing, assesses misfit, and writes out adjoint sources and
        STATIONS_ADJOINT file.

        TODO use concurrent futures to parallelize this

        :type source_name: str
        :param source_name: name of the event to quantify misfit for. If not
            given, will attempt to gather event id from the given task id which
            is assigned by system.run()
        :type save_residuals: str
        :param save_residuals: if not None, path to write misfit/residuls to
        :type save_adjsrcs: str
        :param save_adjsrcs: if not None, path to write adjoint sources to
        :type iteration: int
        :param iteration: current iteration of the workflow, information should
            be provided by `workflow` module if we are running an inversion.
            Defaults to 1 if not given (1st iteration)
        :type step_count: int
        :param step_count: current step count of the line search. Information
            should be provided by the `optimize` module if we are running an
            inversion. Defaults to 0 if not given (1st evaluation)
        :type ntask: int
        :param ntask: number of cpus used for parallel processing
        """
        # GET all source and synthetic names
        if int(source_name) > 1:
            return

        if isinstance(source_names, list):
            ntask = len(source_names)
            observed, synthetic = [], []
            for name in source_names:
                obs_name, syn_name = self._setup_quantify_misfit(name)
                observed.append(obs_name)
                synthetic.append(syn_name)
        elif isinstance(source_name, str):
            observed, synthetic = self._setup_quantify_misfit(source_name)
            observed, synthetic = [observed], [synthetic]
            ntask = 1

        # If the file residual.txt is already exists, delete it first
        if os.path.exists(save_residuals) :
            unix.rm(save_residuals)

        def preprocess_for_parallel(file_idx):

            for obs_fid, syn_fid in zip(observed[file_idx], synthetic[file_idx]):
                obs = self.read(fid=obs_fid)
                syn = self.read(fid=syn_fid)
                # Process observations and synthetics identically
                if self.filter:
                    logger.debug(r"Filtering the obs&syn data")
                    obs = self._apply_filter(obs)
                    syn = self._apply_filter(syn)
                if self.mute:
                    logger.debug(r"Muting the obs&syn data")
                    obs = self._apply_mute(obs)
                    syn = self._apply_mute(syn)
                if self.normalize:
                    logger.debug(r"Normaliziong the obs&syn data")
                    obs = self._apply_normalize(obs)
                    syn = self._apply_normalize(syn)
                
                # Write the residuals/misfit and adjoint sources for each component
                for tr_obs, tr_syn in zip(obs, syn):
                    # logger.debug(r"Saving the residuals the obs&syn data")
                    # Simple check to make sure zip retains ordering
                    assert(tr_obs.stats.component == tr_syn.stats.component)
                    # Calculate the misfit value and write to file
                    if save_residuals and self._calculate_misfit:
                        residual = self._calculate_misfit(
                            obs=tr_obs.data, syn=tr_syn.data,
                            nt=tr_syn.stats.npts, dt=tr_syn.stats.delta
                        )
                        with open(save_residuals, "a") as f:
                            f.write(f"{residual:.2E}\n")

                # New I/O only write once
                # Generate an adjoint source trace, write to file
                save_adjsrcs = True
                if save_adjsrcs and self._generate_adjsrc:
                    logger.debug(r"Calculating and saving the adj data")
                    adjsrc = syn.copy()
                    for tr_obs, tr_syn, tr_adj in zip(obs, syn, adjsrc):
                        #tr_adj.data = tr_syn.data - tr_obs.data
                        tr_adj.data = self._generate_adjsrc(
                                                obs=tr_obs.data, syn=tr_syn.data,
                                                nt=tr_syn.stats.npts, dt=tr_syn.stats.delta
                                            )
                    fid = os.path.basename(syn_fid)
                    fid = self._rename_as_adjoint_source(fid)
                    save_adjsrcs = '/'.join(os.path.dirname(obs_fid).split('/')[:-1])
                    self.write(st=adjsrc, fid=os.path.join(save_adjsrcs, "adj", fid))

        logger.debug("parallel quantifying misfit: ")
        logger.debug(f"misfit: {self.misfit}")
        logger.debug(f"adjoint: {self.adjoint}")
        logger.debug(f"filter: {self.filter}")
        logger.debug(f"normalize: {self.normalize}")
        logger.debug(f"mute: {self.mute}")
        Parallel(n_jobs=nproc)(delayed( preprocess_for_parallel )(idx) for idx in range(ntask))

    def quantify_misfit_line_search(self, source_name=None, source_names=None, save_residuals=None,
                        save_adjsrcs=None, iteration=1, step_count=0, nproc=1,
                        **kwargs):
        """
        Prepares solver for gradient evaluation by writing residuals and
        adjoint traces. Meant to be called by solver.eval_func().

        Reads in observed and synthetic waveforms, applies optional
        preprocessing, assesses misfit, and writes out adjoint sources and
        STATIONS_ADJOINT file.

        TODO use concurrent futures to parallelize this

        :type source_name: str
        :param source_name: name of the event to quantify misfit for. If not
            given, will attempt to gather event id from the given task id which
            is assigned by system.run()
        :type save_residuals: str
        :param save_residuals: if not None, path to write misfit/residuls to
        :type save_adjsrcs: str
        :param save_adjsrcs: if not None, path to write adjoint sources to
        :type iteration: int
        :param iteration: current iteration of the workflow, information should
            be provided by `workflow` module if we are running an inversion.
            Defaults to 1 if not given (1st iteration)
        :type step_count: int
        :param step_count: current step count of the line search. Information
            should be provided by the `optimize` module if we are running an
            inversion. Defaults to 0 if not given (1st evaluation)
        :type ntask: int
        :param ntask: number of cpus used for parallel processing
        """
        # GET all source and synthetic names
        if int(source_name) > 1:
            return

        if isinstance(source_names, list):
            ntask = len(source_names)
            observed, synthetic, adjoint = [], [], []
            for name in source_names:
                obs_name, syn_name, adj_name = self._setup_quantify_misfit_line_search(name)
                observed.append(obs_name)
                synthetic.append(syn_name)
                adjoint.append(syn_name)
        elif isinstance(source_name, str):
            observed, synthetic, adjoint = self._setup_quantify_misfit_line_search(source_name)
            observed, synthetic, adjoint = [observed], [synthetic], [adjoint]
            ntask = 1

        # If the file residual.txt is already exists, delete it first
        if os.path.exists(save_residuals) :
            unix.rm(save_residuals)

        def preprocess_for_parallel(file_idx):

            for obs_fid, syn_fid, adj_fid in zip(observed[file_idx], synthetic[file_idx], adjoint[file_idx]):
                obs = self.read(fid=obs_fid)
                syn = self.read(fid=syn_fid)
                adj = self.read(fid=adj_fid)
                # Process observations and synthetics identically
                if self.filter:
                    logger.debug(r"Filtering the obs&syn data")
                    obs = self._apply_filter(obs)
                    syn = self._apply_filter(syn)
                    adj = self._apply_filter(adj)
                if self.mute:
                    logger.debug(r"Muting the obs&syn data")
                    obs = self._apply_mute(obs)
                    syn = self._apply_mute(syn)
                    adj = self._apply_mute(adj)
                if self.normalize:
                    logger.debug(r"Normaliziong the obs&syn data")
                    obs = self._apply_normalize(obs)
                    syn = self._apply_normalize(syn)
                    adj = self._apply_normalize(adj)
                
                # Write the residuals/misfit and adjoint sources for each component
                for tr_obs, tr_syn, tr_adj in zip(obs, syn, adj):
                    # logger.debug(r"Saving the residuals the obs&syn data")
                    # Simple check to make sure zip retains ordering
                    assert(tr_obs.stats.component == tr_syn.stats.component == tr_adj.stats.component)
                    # Calculate the misfit value and write to file
                    c = tr_adj.data
                    misfit_try = self._generate_adjsrc(
                                            obs=tr_obs.data, syn=tr_syn.data,
                                            nt=tr_syn.stats.npts, dt=tr_adj.stats.delta
                                        )
                    b = c - misfit_try
                    alpha1 = -np.sum(b*c)#-np.dot(b, tr_adj.data)
                    alpha2 = np.sum(b**2)

                    with open(save_residuals, "a") as f:
                        f.write(f"{alpha1:.2E} {alpha2:.2E}\n")

        logger.debug("parallel quantifying misfit for line search: ")
        logger.debug(f"misfit: {self.misfit}")
        logger.debug(f"adjoint: {self.adjoint}")
        logger.debug(f"filter: {self.filter}")
        logger.debug(f"normalize: {self.normalize}")
        logger.debug(f"mute: {self.mute}")

        Parallel(n_jobs=nproc)(delayed( preprocess_for_parallel )(idx) for idx in range(ntask))


    def finalize(self):
        """Teardown procedures for the default preprocessing class"""
        pass

    @staticmethod
    def sum_residuals(residuals):
        """
        Returns the summed square of residuals for each event. Following
        Tape et al. 2007

        :type residuals: np.array
        :param residuals: list of residuals from each NTASK event
        :rtype: float
        :return: sum of squares of residuals
        """
        return np.sum(residuals ** 2.)

    def _apply_filter(self, st):
        """
        Apply a filter to waveform data using ObsPy

        :type st: obspy.core.stream.Stream
        :param st: stream to be filtered
        :rtype: obspy.core.stream.Stream
        :return: filtered traces
        """
        # Pre-processing before filtering
        st.detrend("demean")
        st.detrend("linear")
        st.taper(0.05, type="hann")

        if self.filter.upper() == "BANDPASS":
            st.filter("bandpass", zerophase=True, freqmin=self.min_freq,
                      freqmax=self.max_freq)
        elif self.filter.upper() == "LOWPASS":
            st.filter("lowpass", zerophase=True, freq=self.max_freq)
        elif self.filter.upper() == "HIGHPASS":
            st.filter("highpass", zerophase=True, freq=self.min_freq)

        return st

    def _apply_mute(self, st):
        """
        Apply mute on data based on early or late arrivals, and short or long
        source receiver distances

        .. note::
            The underlying mute functions have been refactored but not tested
            as I was not aware of the intended functionality. Not gauranteed
            to work, use at your own risk.

        :type st: obspy.core.stream.Stream
        :param st: stream to mute
        :rtype: obspy.core.stream.Stream
        :return: muted stream object
        """
        mute_choices = [_.upper() for _ in self.mute]
        if "EARLY" in mute_choices:
            st = signal.mute_arrivals(st, slope=self.early_slope,
                                      const=self.early_const, choice="EARLY")
        if "LATE" in mute_choices:
            st = signal.mute_arrivals(st, slope=self.late_slope,
                                      const=self.late_const, choice="LATE")
        if "SHORT" in mute_choices:
            st = signal.mute_offsets(st, dist=self.short_dist, choice="SHORT")
        if "LONG" in mute_choices:
            st = signal.mute_offsets(st, dist=self.long_dist, choice="LONG")

        return st

    def _apply_normalize(self, st):
        """
        Normalize the amplitudes of waveforms based on user choice

        .. note::
            The normalization function has been refactored but not tested
            as I was not aware of the intended functionality. Not gauranteed
            to work, use at your own risk.

        :type st: obspy.core.stream.Stream
        :param st: All of the data streams to be normalized
        :rtype: obspy.core.stream.Stream
        :return: stream with normalized traces
        """
        st_out = st.copy()
        norm_choices = [_.upper() for _ in self.normalize]

        # Normalize an event by the L1 norm of all traces
        if 'ENORML1' in norm_choices:
            w = 0.
            for tr in st_out:
                w += np.linalg.norm(tr.data, ord=1)
            for tr in st_out:
                tr.data /= w
        # Normalize an event by the L2 norm of all traces
        elif "ENORML2" in norm_choices:
            w = 0.
            for tr in st_out:
                w += np.linalg.norm(tr.data, ord=2)
            for tr in st_out:
                tr.data /= w
        # Normalize each trace by its L1 norm
        if "TNORML1" in norm_choices:
            for tr in st_out:
                w = np.linalg.norm(tr.data, ord=1)
                if w > 0:
                    tr.data /= w
        elif "TNORML2" in norm_choices:
            # normalize each trace by its L2 norm
            for tr in st_out:
                w = np.linalg.norm(tr.data, ord=2)
                if w > 0:
                    tr.data /= w

        return st_out


def read_ascii(fid, origintime=None):
    """
    Read waveforms in two-column ASCII format. This is copied directly from
    pyatoa.utils.read.read_sem()
    """
    try:
        times = np.loadtxt(fname=fid, usecols=0)
        data = np.loadtxt(fname=fid, usecols=1)

    # At some point in 2018, the Specfem developers changed how the ascii files
    # were formatted from two columns to comma separated values, and repeat
    # values represented as 2*value_float where value_float represents the data
    # value as a float
    except ValueError:
        times, data = [], []
        with open(fid, 'r') as f:
            lines = f.readlines()
        for line in lines:
            try:
                time_, data_ = line.strip().split(',')
            except ValueError:
                if "*" in line:
                    time_ = data_ = line.split('*')[-1]
                else:
                    raise ValueError
            times.append(float(time_))
            data.append(float(data_))

        times = np.array(times)
        data = np.array(data)

    if origintime is None:
        origintime = UTCDateTime("1970-01-01T00:00:00")

    # We assume that dt is constant after 'precision' decimal points
    delta = round(times[1] - times[0], 4)

    # Honor that Specfem doesn't start exactly on 0
    origintime += times[0]

    # Write out the header information
    net, sta, cha, fmt = os.path.basename(fid).split('.')
    stats = {"network": net, "station": sta, "location": "",
             "channel": cha, "starttime": origintime, "npts": len(data),
             "delta": delta, "mseed": {"dataquality": 'D'},
             "time_offset": times[0], "format": fmt
             }
    st = Stream([Trace(data=data, header=stats)])

    return st
