"""Perform forward modeling.
"""
import matplotlib.pyplot as plt
import os
import time
import tqdm
import torch
import wavetorch
from mpi4py import MPI
from torch.utils.data import TensorDataset, DataLoader
from wavetorch.utils import ricker_wave, to_tensor

import argparse
import time

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
    
import numpy as np
from sklearn.model_selection import StratifiedKFold

from vowel_helpers import setup_src_coords_customer, setup_probe_coords_customer, get_sources_coordinate_list, setup_probes_moving

parser = argparse.ArgumentParser() 
parser.add_argument('config', type=str, 
                    help='Configuration file for geometry, training, and data preparation')
parser.add_argument('--num_threads', type=int, default=2,
                    help='Number of threads to use')
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA to perform computations')
parser.add_argument('--name', type=str, default=time.strftime('%Y%m%d%H%M%S'),
                    help='Name to use when saving or loading the model file. If not specified when saving a time and date stamp is used')
parser.add_argument('--mode', type=str, default='forward',
                    help='forward modeling or inversion mode')
parser.add_argument('--savedir', type=str, default='./study/',
                    help='Directory in which the model file is saved. Defaults to ./study/')

def refinement(vel, pml_width, fix = 10, water_vel = 1500.):
    nx, nz = vel.shape
    vel_noexpand = vel[pml_width:nx-pml_width, pml_width:nz-pml_width]
    vel_noexpand[0:fix,:] = water_vel
    vel_m = np.pad(vel_noexpand, (pml_width, pml_width), mode='edge')
    return vel_m


if __name__ == '__main__':
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.use_cuda and torch.cuda.is_available():
        args.dev = torch.device('cuda')
    else:
        args.dev = torch.device('cpu')

    'Sets the number of threads used for intraop parallelism on CPU.'
    torch.set_num_threads(args.num_threads)

    print("Configuration: %s" % args.config)
    with open(args.config, 'r') as ymlfile:
        cfg = load(ymlfile, Loader=Loader)

    wavetorch.utils.set_dtype(cfg['dtype'])
    'update_cfg must be called since the width of pml need be added to Nx and Ny'
    cfg = wavetorch.utils.update_cfg(cfg)

    if cfg['seed'] is not None:
        'Sets the seed for generating random numbers. Returns a torch.Generator object.'
        torch.manual_seed(cfg['seed'])
    if cfg['training']['prefix'] is not None:
        args.name = cfg['training']['prefix'] + '_' + args.name
    
    N_classes = len(cfg['data']['vowels'])

            ### Define model
    if cfg['geom']['Nreceivers']>1:
        probes = setup_probe_coords_customer(
                            cfg['geom']['Nreceivers'], cfg['geom']['ipx'], cfg['geom']['py'], cfg['geom']['pd'], 
                            cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N']
                            )
    else:
        probes = []

    source = setup_src_coords_customer(
                        cfg['geom']['src_x'], cfg['geom']['src_y'], cfg['geom']['Nx'],
                        cfg['geom']['Ny'], cfg['geom']['pml']['N']
                        )
    
    'design_region is a array'
    design_region = torch.zeros(cfg['geom']['Ny'], cfg['geom']['Nx'], dtype=torch.uint8)

    VEL_PATH = cfg['geom']['initPath'] if args.mode == 'inversion' else cfg['geom']['cPath']
    geom  = wavetorch.WaveGeometryFreeForm((cfg['geom']['Ny'], cfg['geom']['Nx']), cfg['geom']['h'],  
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
        design_region=design_region
    )

    cell  = wavetorch.WaveCell(cfg['geom']['dt'], geom,
        satdamp_b0=cfg['geom']['nonlinearity']['b0'],
        satdamp_uth=cfg['geom']['nonlinearity']['uth'],
        c_nl=cfg['geom']['nonlinearity']['cnl']
    )

    model = wavetorch.WaveRNN(cell, source, probes)
    model.to(args.dev)

    ### Train
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    criterion = torch.nn.MSELoss()

    ### Get source-x and source-y coordinate in grid cells
    source_x_list, source_y_list = get_sources_coordinate_list(cfg['geom']['ipx'], cfg['geom']['src_y'], cfg['geom']['src_d'], cfg['geom']['Nshots'], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
    
    model.train()
    x = ricker_wave(cfg['geom']['fm'], cfg['geom']['dt'], 2000)
    x = torch.unsqueeze(x, 0)
    if args.mode == 'forward':
        #Records = []
        #for shot in tqdm.trange(cfg['geom']['Nshots']):
        for shot in tqdm.tqdm(range(rank, cfg['geom']['Nshots'], size), position = rank):
            source = setup_src_coords_customer(source_x_list[shot], source_y_list[shot], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
            probe = setup_probes_moving(source_x_list[shot], source_y_list[shot], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
            #print(source, probe)
            model.reset_sources(source)
            model.reset_probes(probe)
            np.save(os.path.join(cfg['geom']['obsPath'], 'shot%03d.npy'%(shot)), model(x).detach().numpy())
            #Records.append(model(x).detach().numpy())
        #Records = np.array(Records)
        #np.save(cfg['geom']['obsPath']+'obs.npy', Records)

    if args.mode == 'inversion':
        ytrue = to_tensor(np.load(cfg['geom']['obsPath']))

        for epoch in tqdm.tqdm(range(0, cfg['training']['N_epochs'])):
            optimizer.zero_grad()
            for shot in range(rank, cfg['geom']['Nshots'], size):
                source = setup_src_coords_customer(source_x_list[shot], source_y_list[shot], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
                probe = setup_probes_moving(source_x_list[shot], source_y_list[shot], cfg['geom']['Nx'], cfg['geom']['Ny'], cfg['geom']['pml']['N'])
                model.reset_sources(source)
                model.reset_probes(probe)
                ypred = model(x)
                loss = criterion(ypred, ytrue[shot])
                loss.backward()

                #np.save(os.path.join(cfg['geom']['savePath'], 'GradShot/Grad%02d_%03d.npy'%(epoch, shot)), model.cell.geom.vel.grad.detach().numpy())

            grad_each_rank = model.cell.geom.vel.grad.detach().numpy()
            grad_root = np.zeros_like(grad_each_rank)
            comm.Allreduce(grad_each_rank, grad_root)
            if rank==0:
                print('current rank:', rank)
                np.save(os.path.join(cfg['geom']['savePath'], 'grad_%02depoch.npy'%(epoch)), grad_root)
                model.cell.geom.vel.grad.data = torch.Tensor(grad_root)
                loss = optimizer.step()
                temp_m = refinement(model.state_dict()['cell.geom.vel'].detach().numpy(), cfg['geom']['pml']['N'])
                model.cell.geom.vel.data = torch.Tensor(temp_m)
                np.save(os.path.join(cfg['geom']['savePath'], 'model_%02depoch.npy'%(epoch)), model.state_dict()['cell.geom.vel'].detach().numpy())
