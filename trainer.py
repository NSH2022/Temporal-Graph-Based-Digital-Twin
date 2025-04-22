import torch
import logging
from torch_geometric.data import Data, Batch
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def train(train_dataloader,NETWORK, BATCH_SIZE, EPOCHS, NET):
    
    logging.info("-----------------------------------------------------------------")
    logging.info(f"Training started.")  
    logging.info("-----------------------------------------------------------------")

    optimizer_inf = torch.optim.Adam(NETWORK.Submodule_inflow.parameters(), lr=1e-3)
    optimizer_tt = torch.optim.Adam(NETWORK.Submodule_travel_time.parameters(), lr=1e-3)
    optimizer_ql = torch.optim.Adam(NETWORK.Submodule_queue_length.parameters(), lr=1e-3)
    optimizer_wt = torch.optim.Adam(NETWORK.Submodule_waiting_time.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    mask, nonzero_indices = get_mask()
    tasks = ['inflow', 'travel time forward', 'travel time reversed', 'queue length', 'waiting time']

    for epoch in range(EPOCHS):
        running_loss_inf = 0.0
        running_loss_tt = 0.0
        running_loss_ql = 0.0
        running_loss_wt = 0.0
        loss_inf, loss_tt, loss_ql, loss_wt = ([] for i in range(4))
        NETWORK.train()

        for idx, batch in enumerate(train_dataloader):
            for item in range(BATCH_SIZE):
                data = batch[item].to(device)

                # extract target variables
                y_tt_fwd, y_tt_rev = data.y["y_tt_E"].squeeze(), data.y["y_tt_W"].squeeze()
                y_tp_fwd, y_tp_rev = data.y["y_tp_E"].squeeze(), data.y["y_tp_W"].squeeze()
                y_ql = data.y["y_ql"]
                y_wt = data.y["y_wt"]
                y_inf = data.x['x_feat'][:,-8:]

                # impute intermediate inflow counts
                optimizer_inf.zero_grad()
                x_inf = data.x['x_feat'][:,-8:].clone()*mask
                out_inf = NETWORK.Submodule_inflow(x_inf, data.edge_index, data.edge_attr)
                loss_inf = criterion(out_inf, y_inf)
                loss_inf.backward()
                optimizer_inf.step()

                # re-use imputed inflow counts
                out_inf_reused = out_inf.detach()
                nonzero_vals = out_inf_reused[nonzero_indices[:,0], nonzero_indices[:,1]]
                data.x['x_feat'][:,-8:][nonzero_indices[:,0], nonzero_indices[:,1]] = nonzero_vals

                # learn bidirectional travel time time series
                optimizer_tt.zero_grad()
                out_tt_fwd, out_tt_rev = NETWORK.Submodule_travel_time(data)
                loss_tt_fwd = criterion(out_tt_fwd, y_tt_fwd)
                loss_tt_rev = criterion(out_tt_rev, y_tt_rev)
                loss_tt = loss_tt_fwd + loss_tt_rev
                loss_tt.backward()
                optimizer_tt.step()

                # learn queue length time series
                optimizer_ql.zero_grad()
                out_ql = NETWORK.Submodule_queue_length(data)
                loss_ql = criterion(out_ql, y_ql)
                loss_ql.backward()
                optimizer_ql.step()

                running_loss_inf += loss_inf.item()
                running_loss_tt += loss_tt.item()
                running_loss_ql += loss_ql.item()

                # learn waiting time time series
                optimizer_wt.zero_grad()
                out_wt = NETWORK.Submodule_waiting_time(data)
                loss_wt = criterion(out_wt, y_wt)
                loss_wt.backward()
                optimizer_wt.step()

                running_loss_inf += loss_inf.item()
                running_loss_tt += loss_tt.item()
                running_loss_wt += loss_wt.item()
        
        logging.info(f'Epoch {epoch}')
        loss_inf = running_loss_inf/ (BATCH_SIZE*len(train_dataloader))
        loss_tt = running_loss_tt/ (BATCH_SIZE*len(train_dataloader))
        loss_ql = running_loss_ql/ (BATCH_SIZE*len(train_dataloader))
        loss_wt = running_loss_wt/ (BATCH_SIZE*len(train_dataloader))
        loss_values = [loss_inf, loss_tt, loss_ql, loss_wt]
                       
                        
        logging.info(f'loss inflow: {loss_inf}')
        logging.info(f'loss travel time: {loss_tt}')
        logging.info(f'loss queue length: {loss_ql}')
        logging.info(f'loss waiting time: {loss_wt}')
        
        loss_task = {k: v for k, v in zip(tasks, loss_values)}
        loss_dic = {f'epoch {epoch}':loss_task}
        save_json(f'models/{NET}_loss.json', loss_dic)



        
def evaluate(test_dataloader, NETWORK, BATCH_SIZE):
    NETWORK.eval()
    true_tt_fwd, pred_tt_fwd, true_tt_rev, pred_tt_rev = ([] for i in range(4))
    true_tp_fwd, pred_tp_fwd, true_tp_rev, pred_tp_rev = ([] for i in range(4))
    true_inf, pred_inf = ([] for i in range(2))
    true_ql, pred_ql = ([] for i in range(2))
    true_wt, pred_wt = ([] for i in range(2))
    sample_batch = next(iter(test_dataloader))

    mask, nonzero_indices = get_mask()
    
    for item in range(BATCH_SIZE):
        data = sample_batch[item].to(device)

        # values of target variables
        y_inf = data.x['x_feat'][:,-8:]
        y_tt_fwd, y_tt_rev = data.y["y_tt_E"], data.y["y_tt_W"]
        y_tp_fwd, y_tp_rev = data.y["y_tp_E"], data.y["y_tp_W"]
        y_ql = data.y["y_ql"]
        y_wt = data.y["y_wt"]


        # impute intermediate inflow counts
        x_inf = data.x['x_feat'][:,-8:].clone()*mask
        out_inf = NETWORK.Submodule_inflow(x_inf, data.edge_index, data.edge_attr)

        # re-use imputed inflow counts
        out_inf_reused = out_inf.detach()
        nonzero_vals = out_inf_reused[nonzero_indices[:,0], nonzero_indices[:,1]]
        data.x['x_feat'][:,-8:][nonzero_indices[:,0], nonzero_indices[:,1]] = nonzero_vals

        # model outputs
        out_tt_fwd, out_tt_rev, out_ql, out_wt = NETWORK(data)

        pred_inf.append(out_inf.detach().cpu())
        true_inf.append(y_inf.cpu())
        pred_tt_fwd.append(out_tt_fwd.squeeze().detach().cpu())
        true_tt_fwd.append(y_tt_fwd.cpu())
        pred_tt_rev.append(out_tt_rev.squeeze().detach().cpu())
        true_tt_rev.append(y_tt_rev.cpu())
        pred_ql.append(out_ql.detach().cpu())
        true_ql.append(y_ql.cpu())
        pred_wt.append(out_wt.detach().cpu())
        true_wt.append(y_wt.cpu())

    out_inf = torch.stack(true_inf), torch.stack(pred_inf)
    out_tt_fwd = torch.stack(true_tt_fwd), torch.stack(pred_tt_fwd)
    out_tt_rev = torch.stack(true_tt_rev), torch.stack(pred_tt_rev)
    out_ql = torch.stack(true_ql), torch.stack(pred_ql)
    out_wt =torch.stack(true_wt), torch.stack(pred_wt)
    
    return out_inf, out_tt_fwd, out_tt_rev, out_ql, out_wt
    