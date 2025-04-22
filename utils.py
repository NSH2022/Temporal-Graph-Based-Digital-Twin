import pickle
import random 
import torch
import matplotlib.pyplot as plt
from importlib import reload
plt = reload(plt)
import numpy as np
import logging
import json
import os
from torch.utils.data import Subset
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 


def load(fileName):
    with open(fileName + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def save(file, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)


def save_json(json_pth, data):
    if os.path.exists(json_pth):
        with open(json_pth, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.update(data)
        data = existing_data
        
    # Save the updated data back to the JSON file
    with open(json_pth, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def train_val_dataset(dataset, val_split=0.20):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def get_mask():
    inter_mask = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]).unsqueeze(0).repeat(6, 1)
    dege_mask = torch.ones(1,8)
    mask = torch.cat((dege_mask, inter_mask, dege_mask), dim=0).to(device)
    nonzero_indices = torch.nonzero(mask == 0, as_tuple=False)
    return mask, nonzero_indices





def compute_error_metrics_1(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: str = 'mean'):

    # Ensure inputs are float tensors
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Compute differences
    diff = y_true - y_pred
    
    # Mean Absolute Error (MAE)
    mae = torch.abs(diff)
    mae = mae.mean() if reduction == 'mean' else mae.sum() if reduction == 'sum' else mae
    
    # Mean Squared Error (MSE)
    mse = diff.pow(2)
    mse = mse.mean() if reduction == 'mean' else mse.sum() if reduction == 'sum' else mse
    
    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(mse) if reduction != 'none' else torch.sqrt(mse)
    
    # Normalized Root Mean Squared Error (NRMSE)
    range_true = y_true.max() - y_true.min()
    nrmse = rmse / (range_true + 1e-8)  # Add epsilon for stability

    ci = 1.96 * torch.sqrt(mse)
    
    return (mae.item(), mse.item(), rmse.item(), nrmse.item())




def compute_error_metrics_2(q: torch.Tensor, p: torch.Tensor):

    # Flatten to 2D (batch_size, -1)
    pred = p.view(p.size(0), -1)
    true = q.view(q.size(0), -1)

    # Normalize each sample to sum to 1
    pred = pred / (pred.sum(dim=1, keepdim=True) + 1e-8)
    true = true / (true.sum(dim=1, keepdim=True) + 1e-8)
    

    # MAPE
    mape = abs((np.mean(p.numpy()) - np.mean(q.numpy())) / np.mean(q.numpy()))

    # Wasserstein distance or Earth Mover's Distance (EMD)
    emd = wasserstein_distance(p.view(-1), q.view(-1))

    # Hellinger Distance (per sample, then average)
    hld = torch.sqrt(torch.sum((torch.sqrt(pred) - torch.sqrt(true))**2, dim=1)) / torch.sqrt(torch.tensor(2.0))
    hld = hld.mean()

    # NRMSE
    mse = F.mse_loss(pred, true)
    rmse = torch.sqrt(mse)
    nrmse = rmse / (true.max() - true.min() + 1e-8)


    return (mape.item(), emd.item(), hld.item(), nrmse.item())


    
          
          

def compute_errors(output):
    out_inf, out_tt_fwd, out_tt_rev, out_ql, out_wt = output
    tasks = ['inflow', 'travel time forward', 'travel time reversed', 'queue length', 'waiting time']
    error_measures_1 = ['MAE', 'MSE', 'RMSE', 'NRMSE']
    error_measures_2 = ['MAPE', 'EMD', 'HLD', 'NRMSE']
    error_values_inf = {k: v for k, v in zip(error_measures_1, compute_error_metrics_1(out_inf[0], out_inf[1]))}
    error_values_tt_fwd = {k: v for k, v in zip(error_measures_2, compute_error_metrics_2(out_tt_fwd[0], out_tt_fwd[1]))}
    error_values_tt_rev = {k: v for k, v in zip(error_measures_2, compute_error_metrics_2(out_tt_rev[0], out_tt_rev[1]))}
    error_values_ql = {k: v for k, v in zip(error_measures_1, compute_error_metrics_1(out_ql[0], out_ql[1]))}
    error_values_wt = {k: v for k, v in zip(error_measures_2, compute_error_metrics_2(out_wt[0], out_wt[1]))}
    error_values = [error_values_inf, error_values_tt_fwd, error_values_tt_fwd, error_values_ql, error_values_wt]
    error_task = {k: v for k, v in zip(tasks, error_values)}

    save_json(f'results/total_error_values.json', error_task)
    logging.info('Error values are computed.')

    
def plot_inflow(output, BATCH_SIZE):
    true_inf, pred_inf = output
    
    plt.figure(figsize=(8, 6))
    x1 = torch.tensor(list(range(0, BATCH_SIZE)))[:100]
    selected_isc = np.random.randint(0, 8) # intersection J1, ...J9
    selected_lane = np.random.choice([0, 1, 4, 5]) # major phases: 1, 2, 5, 6

    plt.rcParams.update({'font.size': 18})
    plt.plot(x1, true_inf[:100,selected_isc,selected_lane] , label="Actual", color="red", linewidth=4, marker = '*')
    plt.plot(x1, pred_inf[:100,selected_isc,selected_lane], label="Predicted", color="green", linewidth=4)

    plt.title('Square Number',fontsize=24)
    plt.xlabel("Sample ID")
    plt.ylabel("Vehicle Counts")
    plt.title(f"Inflow Traffic Volume : J{selected_isc}-P{selected_lane}")
    plt.legend()


    plt.savefig(
        'results/inflow_counts.jpg',        
        dpi=300,                
        bbox_inches='tight',    
        pad_inches=0.1,         
        transparent=True)


        
def plot_travel_time_reversed(output, BATCH_SIZE):
    true_tt_rev, pred_tt_rev = output
    
    plt.figure(figsize=(8, 6))
    selected_sample = np.random.choice(BATCH_SIZE) # major phases: 1, 2, 5, 6
    x1 = list(range(0, 50, 5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(x1, true_tt_rev[selected_sample,:], label="Actual", color="red", linewidth=4)
    plt.plot(x1, pred_tt_rev[selected_sample, :], label="Predicted", color="green", linewidth=4)


    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Travel Time (Seconds)")
    plt.title(f"Westbound Corridor Travel Time")
    plt.legend()


    plt.savefig(
        'results/westbound_traveltime.jpg',        
        dpi=300,                
        bbox_inches='tight',    
        pad_inches=0.1,         
        transparent=True)
    logging.info('Travel time Plot is created.')


          
        
def plot_travel_time_forward(output, BATCH_SIZE):
    true_tt_fwd, pred_tt_fwd = output
    plt.figure(figsize=(8, 6))
    selected_sample = np.random.choice(BATCH_SIZE) # major phases: 1, 2, 5, 6
    x1 = list(range(0, 50, 5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(x1, true_tt_fwd[selected_sample,:], label="Actual", color="red", linewidth=4)
    plt.plot(x1, pred_tt_fwd[selected_sample, :], label="Predicted", color="green", linewidth=4)


    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Travel Time (Seconds)")
    plt.title(f"Eastbound Corridor Travel Time")
    plt.legend()


    plt.savefig(
        'results/eastbound_traveltime.jpg', 
        dpi=300,               
        bbox_inches='tight',    
        pad_inches=0.1,        
        transparent=True       
    )
    logging.info("Plots are created.")
    
        
def plot_queue_length(output, BATCH_SIZE):
    true_ql, pred_ql = output
    plt.figure(figsize=(8, 6))
    selected_sample = np.random.choice(BATCH_SIZE) # major phases: 1, 2, 5, 6
    selected_intersect = np.random.choice(8)
    selected_phase = np.random.choice(8)
    x1 = list(range(0, 50, 5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(x1, true_ql[selected_sample,selected_intersect, selected_phase], label="Actual", color="red", linewidth=4)
    plt.plot(x1, pred_ql[selected_sample,selected_intersect, selected_phase], label="Predicted", color="green", linewidth=4)


    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Queue Length (Meters)")
    plt.title(f"Maximum Queue Lengths : J{selected_intersect+1}-P{selected_phase+1}")
    plt.legend()


    plt.savefig(
        'results/queue_length.jpg',       
        dpi=300,               
        bbox_inches='tight',   
        pad_inches=0.1,       
        transparent=True        
    )

    
    
    
        
def plot_travel_waiting_time(output, BATCH_SIZE):
    true_wt, pred_wt= output
    plt.figure(figsize=(8, 6))
    selected_sample = np.random.choice(BATCH_SIZE) # major phases: 1, 2, 5, 6
    selected_intersect = np.random.choice(8)
    selected_phase = np.random.choice(8)
    x1 = list(range(0, 50, 5))
    plt.rcParams.update({'font.size': 18})
    plt.plot(x1, true_wt[selected_sample,selected_intersect, selected_phase], label="Actual", color="red", linewidth=4)
    plt.plot(x1, pred_wt[selected_sample,selected_intersect, selected_phase], label="Predicted", color="green", linewidth=4)


    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Waiting Time (Seconds)")
    plt.title(f"Waiting Time : J{selected_intersect+1}-P{selected_phase+1}")
    plt.legend()


    plt.savefig(
        'results/waiting_time.jpg',        
        dpi=300,                
        bbox_inches='tight',    
        pad_inches=0.1,        
        transparent=True        
    )
    logging.info("Plots are created.")
                        