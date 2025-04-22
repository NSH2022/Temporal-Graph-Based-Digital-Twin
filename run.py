import torch
from torch_geometric.loader import dataloader as DL
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import logging
import argparse
import warnings
import os
import random

from dataset import Graph_Dataset
from nets import DTGAT
from trainer import train, evaluate
from utils import *

torch.set_float32_matmul_precision('medium')
torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--NET', default = 'TGDT')
    args = parser.parse_args()
    NET = args.NET
    N_INTERSECT = 9
    BATCH_SIZE= 10
    EPOCHS = 20
    NO_CPUs = 20 # change according to your available resources
    NO_GPUs = 2 # change according to your available resources

    NODE_NAME = load('helperfiles/NODE_NAME_Corridor')
    PROCESS_PATH='data/'
    MODEL_PATH = f'models/{NET}'
    


    ########### load data #################
    exp_dataset = Graph_Dataset(PROCESS_PATH)
    dataset = train_val_dataset(exp_dataset, val_split= 0.10)
    train_dataloader = DL.DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers = NO_CPUs, drop_last = True)
    test_dataloader = DL.DataLoader(dataset['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers = NO_CPUs, drop_last = True)

    logging.info(f"Dataset: SUMO-REAL | Samples: {len(exp_dataset)}| Intersections:{N_INTERSECT}")
    logging.info(f"Number of batches: Train: {len(train_dataloader)}| Test: {len(test_dataloader)}, Model: {NET}")


    
    
    
    
    ########### train the model #################
    NETWORK = DTGAT().to(device)
    train(train_dataloader, NETWORK, BATCH_SIZE, EPOCHS, NET)
    torch.save(NETWORK.state_dict(), MODEL_PATH)

    
    ############# evaluate the results ############
    LOADED_NETWORK = DTGAT().to(device)
    LOADED_NETWORK.load_state_dict(torch.load(MODEL_PATH))
    LOADED_NETWORK.to(device)

    output = evaluate(test_dataloader, LOADED_NETWORK, BATCH_SIZE)
    compute_errors(output)


    # ############ plot estimated vs actual variables ########
    out_inf, out_tt_fwd, out_tt_rev, out_ql, out_wt = output
    plot_inflow(out_inf, BATCH_SIZE)
    plot_travel_time_reversed(out_tt_rev, BATCH_SIZE)
    plot_travel_time_forward(out_tt_fwd, BATCH_SIZE)
    # plot_queue_length(out_ql, BATCH_SIZE)
    # plot_travel_waiting_time(out_wt, BATCH_SIZE)
    


   

    
 
    
    


 
