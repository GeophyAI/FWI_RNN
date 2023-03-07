"""Perform forward modeling.
"""
import matplotlib.pyplot as plt
import os
import time
import tqdm
import torch

import sys
sys.path.append("..")
sys.path.append("./KFAC")
import wavetorch
from mpi4py import MPI
from torch.utils.data import TensorDataset, DataLoader
from wavetorch.utils import ricker_wave, to_tensor, cpu_fft, CNN
from KFAC.optimizers import KFACOptimizer


import argparse
import time
from copy import deepcopy
from joblib import Parallel, delayed

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
    
import numpy as np
from sklearn.model_selection import StratifiedKFold
from vowel_helpers import setup_src_coords_customer, setup_probe_coords_customer, get_sources_coordinate_list

parser = argparse.ArgumentParser() 
parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('--num_threads', type=int, default=2,
                    help='Number of threads to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
parser.add_argument('-optimizer', choices=['adam'], default='adam',
                    help='optimizer (adam)')
parser.add_argument('--mode', type=str, default='forward',
                    help='forward modeling or inversion mode')
parser.add_argument('--savedir', type=str, default='./study/',
                    help='Directory in which the model file is saved. Defaults to ./study/')




if __name__ == '__main__':
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.use_cuda and torch.cuda.is_available():
        args.dev = torch.device('cuda')
        print("Using CUDA for calculation")
    else:
        args.dev = torch.device('cpu')
        print("Using CPU for calculation")

    'Sets the number of threads used for intraop parallelism on CPU.'
    torch.set_num_threads(args.num_threads)

    print("Configuration: %s" % args.config)
    with open(args.config, 'r') as ymlfile:
        cfg = load(ymlfile, Loader=Loader)

    wavetorch.utils.set_dtype(cfg['dtype'])
    'update_cfg must be called since the width of pml need be added to Nx and Ny'
    ori_cfg = deepcopy(cfg)
    cfg = wavetorch.utils.update_cfg(cfg)

    if cfg['seed'] is not None:
        'Sets the seed for generating random numbers. Returns a torch.Generator object.'
        torch.manual_seed(cfg['seed'])
    if cfg['training']['prefix'] is not None:
        args.name = cfg['training']['prefix'] + '_' + args.name

            ### Define model
    probes = setup_probe_coords_customer(
                        cfg['geom']['Nreceivers'], cfg['geom']['ipx'], cfg['geom']['py'], cfg['geom']['pd'], 
                        cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N']
                        )

    source = setup_src_coords_customer(
                        cfg['geom']['src_x'], cfg['geom']['src_y'], cfg['geom']['Nx'],
                        cfg['geom']['Ny'], cfg['geom']['pml']['N']
                        )
    
    'design_region is a array'
    design_region = torch.zeros(cfg['geom']['Ny'], cfg['geom']['Nx'], dtype=torch.uint8)

    VEL_PATH = cfg['geom']['initPath'] if args.mode == 'inversion' else cfg['geom']['cPath']
    geom  = wavetorch.WaveGeometryFreeForm(
        domain_shape = (cfg['geom']['Ny'], cfg['geom']['Nx']), 
        h = cfg['geom']['h'],  
        cPath = VEL_PATH,
        c0=cfg['geom']['c0'], 
        c1=cfg['geom']['c1'],
        eta=cfg['geom']['binarization']['eta'],
        beta=cfg['geom']['binarization']['beta'],
        abs_sig=cfg['geom']['pml']['max'], 
        abs_N=cfg['geom']['pml']['N'], 
        abs_p=cfg['geom']['pml']['p'],
        rho=cfg['geom']['init'],
        blur_radius=cfg['geom']['blur_radius'],
        blur_N=cfg['geom']['blur_N'],
        design_region=design_region, 
        device = args.dev
    )

    cell  = wavetorch.WaveCell(cfg['geom']['dt'], geom,
        satdamp_b0=cfg['geom']['nonlinearity']['b0'],
        satdamp_uth=cfg['geom']['nonlinearity']['uth'],
        c_nl=cfg['geom']['nonlinearity']['cnl']
    )
    
    model = wavetorch.WaveRNN(cell, source, probes)
    cnn = CNN()
    model.to(args.dev)
    cnn.to(args.dev)

    ### Train
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'], eps=1e-12)


    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=cfg['training']['lr'], eps=1e-12)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=cfg['training']['lr'], history_size=10, line_search_fn='strong_wolfe')
    criterion = torch.nn.MSELoss()
    criterion_cnn = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()

    ### Get source-x and source-y coordinate in grid cells
    source_x_list, source_y_list = get_sources_coordinate_list(cfg['geom']['ipx'], cfg['geom']['src_y'], cfg['geom']['src_d'], cfg['geom']['Nshots'], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
    
    model.train()
    #cnn.train()

    #x = torch.unsqueeze(x, 0)

    record = np.zeros((cfg['geom']['Nshots'], cfg['geom']['nt'], ori_cfg['geom']['Nx'], 1), dtype=np.float32)
    if args.mode == 'forward':
        x = ricker_wave(cfg['geom']['fm'], cfg['geom']['dt'], cfg['geom']['nt'])
        x = torch.unsqueeze(x, 0)
        for shot in tqdm.tqdm(range(rank, cfg['geom']['Nshots'], size), position = rank):
            source = setup_src_coords_customer(source_x_list[shot], source_y_list[shot], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
            model.reset_sources(source)
            record[shot] = model(x).cpu().detach().numpy()

        # Save the records
        np.save(cfg['geom']['obsPath'], record)


    if args.mode == 'inversion':

        loss_cpu = np.zeros((len(cfg['geom']['multiscale']), cfg['training']['N_epochs'], cfg['geom']['Nshots']), np.float32)

        x = ricker_wave(cfg['geom']['fm'], cfg['geom']['dt'], cfg['geom']['nt'], dtype='numpy')
        full_band_data = np.load(cfg['geom']['obsPath'])
        filtered_data = np.zeros_like(full_band_data)

        for idx_freq, freq in enumerate(cfg['geom']['multiscale']):

            # Filter both record and ricker
            filtered_data = cpu_fft(full_band_data.copy(), cfg['geom']['dt'], N=1, low=freq, axis = 1, mode='lowpass')
            filtered_ricker = cpu_fft(x.copy(), cfg['geom']['dt'], N=1, low=freq, axis = 0, mode='lowpass')
            filtered_ricker = torch.unsqueeze(torch.from_numpy(filtered_ricker), 0)
            ytrue = to_tensor(filtered_data)

            pbar = tqdm.tqdm(range(0, cfg['training']['N_epochs']), leave=True)
            for epoch in pbar:

                optimizer.zero_grad()

                if cfg['training']['sample_shot']:
                    shots_current_epoch = np.random.choice(np.arange(cfg['geom']['Nshots']), cfg['training']['shot_per_epoch'], replace=True)
                    shot_range = range(rank, cfg['training']['shot_per_epoch'], size)
                else:
                    shots_current_epoch = np.arange(cfg['geom']['Nshots'])
                    shot_range = range(rank, cfg['geom']['Nshots'], size)

                for shot in shot_range:
                    shot = shots_current_epoch[shot]
                    source = setup_src_coords_customer(source_x_list[shot], source_y_list[shot], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
                    model.reset_sources(source)

                    ypred = model(filtered_ricker)
                    loss = criterion(ypred, ytrue[shot].to(ypred.device).unsqueeze(0))
                    loss_cpu[idx_freq, epoch, shot] = loss.cpu().detach().numpy()

                    #Precondition with Hessian
                    PRECOND = False
                    if PRECOND:
                        env_grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
                        length = cfg['geom']['Nx'] * cfg['geom']['Ny']
                        HESSIAN = np.zeros(shape=(length, length), dtype=np.float32)
                        hess_params = torch.zeros_like(env_grads[0])

                        for i in tqdm.trange(env_grads[0].size(0)):
                            for j in tqdm.trange(env_grads[0].size(1)):
                                hess_params = torch.autograd.grad(env_grads[0][i][j], model.parameters(), retain_graph=True)[0]
                                HESSIAN[i*env_grads[0].size(1)+j] = hess_params.detach().cpu().numpy().flatten()

                        # Save the Hessian
                        np.save("/mnt/others/DATA/Inversion/RNN_Hessian/hessian/hessian_%shot.npy"%(shot), HESSIAN)

                    loss.backward()
                    #np.save("/mnt/others/DATA/Inversion/RNN_Hessian/grad/grad_%shot.npy"%(shot), model.cell.geom.vel.grad.cpu().detach().numpy())

                #pbar.set_description(f'Freq [%d/%d]'%(idx_freq+1, len(cfg['geom']['multiscale'])))
                #grad_each_rank = model.cell.geom.vel.grad.cpu().detach().numpy()
                #grad_root = np.zeros_like(grad_each_rank)
                #comm.Allreduce(grad_each_rank, grad_root)
                if rank==0:

                    #np.save(os.path.join(cfg['geom']['inv_savePath'], 'grad_%02depoch.npy'%(epoch)), grad_root)

                    loss = optimizer.step()

                    vel = model.cell.geom.vel.cpu().detach().numpy().squeeze()
                    # Reset water
                    # vel[0:cfg['geom']['pml']['N']+24,:] = 1500
                    # model.cell.geom.c.data = to_tensor(vel).to(args.dev)

                    #perturbation = np.random.random(vel.shape).astype(vel.dtype)*5
                    #vel+=perturbation
                    #model.cell.geom.c = vel
                    np.save(os.path.join(cfg['geom']['inv_savePath'], 'vel%.2ffreq_%02depoch.npy'%(freq, epoch)), vel)
                    #np.save(os.path.join(cfg['geom']['inv_savePath'], 'loss.npy'), loss_cpu)

