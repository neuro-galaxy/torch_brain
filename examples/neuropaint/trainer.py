import torch
import numpy as np
import wandb
import os
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2, torch_corrcoef
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Trainer():
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            consistency,
            encoder_stitcher_ema,
            **kwargs
    ):
        # get all the arguments
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

        # get arguments from kwargs if they exist
        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)
        self.multi_gpu = kwargs.get("multi_gpu", False)
        self.with_reg = kwargs.get("with_reg", False)

        self.metric = 'r2'
        self.session_active_neurons = []
        self.masking_mode = self.config.model.encoder.masker.mask_mode

        self.eids = kwargs['eids']

        
        self.consistency = consistency
        self.encoder_stitcher_ema = encoder_stitcher_ema
        self.global_step = 0

        num_factors = self.config.model.encoder.stitcher.n_channels_per_region
        areaoi_ind = kwargs.get("areaoi_ind", None)
        num_areas = len(areaoi_ind)
        target_kernel_each_type = {frozenset({areaoi_ind[i], areaoi_ind[j]}): {'kernel':torch.ones(num_factors, num_factors).to(self.accelerator.device), 'count':0} for i in range(num_areas) for j in range(i, num_areas)}
        self.target_kernel_stored = {'left': target_kernel_each_type, 'right': target_kernel_each_type} 

    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
        # train loop
        for epoch in range(self.config.training.num_epochs):
            
            self.train_dataloader.set_epoch(epoch)
            self.eval_dataloader.set_epoch(epoch)

            train_epoch_results = self.train_epoch()
            eval_epoch_results = self.eval_epoch()
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss'] }")

            if eval_epoch_results:
                if eval_epoch_results[f'eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss}")
                    print(f"epoch: {epoch} eval trial avg {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")
                    # save model
                    if self.accelerator.is_local_main_process and self.accelerator.process_index == 0:
                        self.save_model(name="best_eval_loss", epoch=epoch)

                    if self.config.wandb.use and self.accelerator.is_local_main_process and self.accelerator.process_index == 0:
                        gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0], 
                        preds=eval_epoch_results['eval_preds'][0], epoch=epoch,
                        active_neurons=self.session_active_neurons[0][:5])

                        wandb.log({"best_epoch": epoch,
                                "best_gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                                "best_r2_fig": wandb.Image(gt_pred_fig['plot_r2'])})
                
                if eval_epoch_results[f'eval_trial_avg_{self.metric}']> best_eval_trial_avg_metric:
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval trial avg {self.metric}: {best_eval_trial_avg_metric}")

                    if self.config.wandb.use and self.accelerator.is_local_main_process and self.accelerator.process_index == 0:
                        self.save_model(name="best_r2_active_neurons", epoch=epoch)
                
                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

            # save model by epoch
            if self.accelerator.is_local_main_process and self.accelerator.process_index == 0:
                if epoch % self.config.training.save_every == 0:
                    self.save_model(name="epoch", epoch=epoch)

            # plot epoch
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                if self.config.wandb.use and self.accelerator.is_local_main_process and self.accelerator.process_index == 0:
                        gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0], 
                        preds=eval_epoch_results['eval_preds'][0], 
                        epoch=epoch,
                        active_neurons=self.session_active_neurons[0][:5])
                        wandb.log({
                            "gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                            "r2_fig": wandb.Image(gt_pred_fig['plot_r2'])
                        })

            # wandb log
            if self.config.wandb.use and self.accelerator.is_local_main_process and self.accelerator.process_index == 0:

                if self.consistency:
                    wandb.log({
                        "train_loss": train_epoch_results['train_loss'],
                        "train_regularization_loss": train_epoch_results['train_regularization_loss'],
                        "train_consistency_loss": train_epoch_results['train_consistency_loss'],
                        "eval_loss": eval_epoch_results['eval_loss'],
                        "eval_regularization_loss": eval_epoch_results['eval_regularization_loss'],
                        "eval_consistency_loss": eval_epoch_results['eval_consistency_loss'],
                        f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    })
                else:
                    wandb.log({
                        "train_loss": train_epoch_results['train_loss'],
                        "train_regularization_loss": train_epoch_results['train_regularization_loss'],
                        "eval_loss": eval_epoch_results['eval_loss'],
                        "eval_regularization_loss": eval_epoch_results['eval_regularization_loss'],
                        f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    })
                
        # save last model
        if self.accelerator.is_local_main_process and self.accelerator.process_index == 0:
            self.save_model(name="last", epoch=epoch)

            
    def train_epoch(self):
        train_loss = 0.
        train_regularization_loss = 0.
        self.model.train()
        
        if self.consistency:
            self.encoder_stitcher_ema.train()
            train_consistency_loss = 0.

        for batch in tqdm(self.train_dataloader):
            self.global_step += 1
            batch = move_batch_to_device(batch, self.accelerator.device)
            
            outputs = self._forward_model_outputs(batch, self.masking_mode)
            loss = outputs.loss

            if self.consistency:     
                consistency_loss = self._compute_consistency_loss(batch)
                loss += consistency_loss
                train_consistency_loss += consistency_loss

            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.consistency:      
                self._update_ema_variables(alpha=0.999)
                self._update_target_kernel(batch)

            train_loss += loss.item()
            train_regularization_loss += outputs.regularization_loss.item()

        if self.consistency:
            return{
                "train_loss": train_loss/len(self.train_dataloader),
                "train_consistency_loss": train_consistency_loss.item()/len(self.train_dataloader),
                "train_regularization_loss": train_regularization_loss/len(self.train_dataloader)
            }
        else:
            return{
                "train_loss": train_loss/len(self.train_dataloader),
                "train_regularization_loss": train_regularization_loss/len(self.train_dataloader)
            }
        
    def _forward_model_outputs(self, batch, masking_mode):

        return self.model(
            batch['spikes_data'], 
            spikes_timestamps=batch['spikes_timestamps'], 
            neuron_regions=batch['neuron_regions'],
            is_left = batch['is_left'],
            trial_type=batch['trial_type'],
            masking_mode=masking_mode, 
            eid=batch['eid'][0],  # each batch consists of data from the same eid
            with_reg=self.with_reg
        ) 
    
    def _update_ema_variables(self, alpha=0.999):

        alpha = min(1 - 1 / (self.global_step + 1), alpha)

        with torch.no_grad():
            model_encoder = self.model.module.encoder if self.multi_gpu else self.model.encoder
            for ema_param, param in zip(self.encoder_stitcher_ema.parameters(), model_encoder.stitchers.parameters()):
                ema_param.data = alpha * ema_param.data + (1. - alpha) * param.data


    def _update_target_kernel(self, batch):
        # x is (B, T, R, n_channels_per_region)
        x = self.encoder_stitcher_ema(batch['spikes_data'], str(batch['eid'][0].item()), batch['neuron_regions'], batch['is_left'])
        area_ind_unique = batch['neuron_regions'][0].unique()

        for i in range(len(area_ind_unique)):
            area_i = area_ind_unique[i].item()
            for j in range(i, len(area_ind_unique)):
                area_j = area_ind_unique[j].item()

                for trial_type in ['left', 'right']:

                    if trial_type == 'left':
                        trial_type_flag = (batch['trial_type'] == -1)
                    elif trial_type == 'right':
                        trial_type_flag = (batch['trial_type'] == 1)

                    trial_num = trial_type_flag.sum()

                    if trial_num.item() < 5:
                        continue    
                
                    x_i = x[trial_type_flag,:,i,:].view(-1, x.shape[-1]) # (B*T, n_channels_per_region)
                    x_j = x[trial_type_flag,:,j,:].view(-1, x.shape[-1]) 

                    kernel = torch_corrcoef(x_i, x_j) # (n_channels_per_region, n_channels_per_region)
                    
                    key_tmp = frozenset({area_i, area_j})
                    stored_kernel = self.target_kernel_stored[trial_type][key_tmp]['kernel']
                    count = self.target_kernel_stored[trial_type][key_tmp]['count']
                    self.target_kernel_stored[trial_type][key_tmp]['kernel'] = (stored_kernel * count + kernel) / (count + 1)
                    self.target_kernel_stored[trial_type][key_tmp]['count'] += 1
        
    def _compute_consistency_loss(self, batch):
        model_encoder = self.model.module.encoder if self.multi_gpu else self.model.encoder
        x = model_encoder.stitchers(batch['spikes_data'], str(batch['eid'][0].item()), batch['neuron_regions'], batch['is_left'])
        area_ind_unique = batch['neuron_regions'][0].unique()

        consistency_loss_tmp = 0
        kernel_num = 0

        cos = torch.nn.CosineSimilarity(dim=0)
        triu_indices = torch.triu_indices(x.shape[-1], x.shape[-1], offset=1)

        for i in range(len(area_ind_unique)):
            area_i = area_ind_unique[i].item()
            for j in range(i, len(area_ind_unique)):
                area_j = area_ind_unique[j].item()

                for trial_type in ['left', 'right']:

                    if trial_type == 'left':
                        trial_type_flag = (batch['trial_type'] == -1)
                    elif trial_type == 'right':
                        trial_type_flag = (batch['trial_type'] == 1)

                    trial_num = trial_type_flag.sum()

                    if trial_num.item() < 5:
                        continue 

                    x_i = x[trial_type_flag,:,i,:].view(-1, x.shape[-1])
                    x_j = x[trial_type_flag,:,j,:].view(-1, x.shape[-1])

                    kernel = torch_corrcoef(x_i, x_j)
                    key_tmp = frozenset({area_i, area_j})
                    target_kernel = self.target_kernel_stored[trial_type][key_tmp]['kernel']
                    if j==i:
                        consistency_loss_tmp += 1 - cos(kernel[triu_indices[0], triu_indices[1]], target_kernel[triu_indices[0], triu_indices[1]])
                    else:
                        consistency_loss_tmp += 1 - cos(kernel.view(-1), target_kernel.view(-1))
                    kernel_num += 1

        return consistency_loss_tmp/(kernel_num + 1e-8)

    def eval_epoch(self): 
        self.model.eval()
        if self.consistency:
            self.encoder_stitcher_ema.eval()
            eval_consistency_loss = 0.

        eval_loss = 0.
        eval_regularization_loss = 0.
        
        session_results = {}
        for eid in self.eids:
            session_results[eid] = {
                "gt": [],
                "preds": []
            }
        if self.eval_dataloader:
            gt, preds = [], []
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    batch = move_batch_to_device(batch, self.accelerator.device)
                    outputs = self._forward_model_outputs(batch, self.masking_mode)
                    loss = outputs.loss

                    if self.consistency:
                        consistency_loss = self._compute_consistency_loss(batch)
                        loss += consistency_loss
                        eval_consistency_loss += consistency_loss

                    eval_loss += loss.item()

                    eid = batch['eid'][0].item()
                    session_results[eid]["gt"].append(outputs.targets.clone())
                    session_results[eid]["preds"].append(outputs.preds.clone())

                    eval_regularization_loss += outputs.regularization_loss.item()
                    
            results_list = []
            for idx, eid in enumerate(self.eids):
                _gt = torch.cat(session_results[eid]["gt"], dim=0)
                _preds = torch.cat(session_results[eid]["preds"], dim=0)
                
                _preds = torch.exp(_preds)
                gt.append(_gt)
                preds.append(_preds)

                if len(self.session_active_neurons) < len(self.eids):
                    active_neurons = np.argsort(gt[idx].cpu().numpy().sum((0,1)))[::-1][:50].tolist()
                    self.session_active_neurons.append(active_neurons)

                results = metrics_list(gt = gt[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0),
                                       pred = preds[idx][:,:,self.session_active_neurons[idx]].transpose(-1,0), 
                                       metrics=["r2"],
                                       device=self.accelerator.device)
                
                results_list.append(results[self.metric])

        if self.consistency:
            return {
                "eval_loss": eval_loss/len(self.eval_dataloader),
                "eval_regularization_loss": eval_regularization_loss/len(self.eval_dataloader),
                "eval_consistency_loss": eval_consistency_loss.item()/len(self.eval_dataloader),
                f"eval_trial_avg_{self.metric}": np.mean(results_list),
                "eval_gt": gt,
                "eval_preds": preds,
            }
        else:
            return {
                "eval_loss": eval_loss/len(self.eval_dataloader),
                "eval_regularization_loss": eval_regularization_loss/len(self.eval_dataloader),
                f"eval_trial_avg_{self.metric}": np.mean(results_list),
                "eval_gt": gt,
                "eval_preds": preds,
            }
    
    def plot_epoch(self, gt, preds, epoch, active_neurons):
        gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().numpy(),
                    pred = preds.mean(0).T.detach().cpu().numpy(),
                    epoch = epoch)
        
        r2_fig = plot_neurons_r2(gt = gt.mean(0),
                pred = preds.mean(0),
                neuron_idx=active_neurons,
                epoch = epoch)
        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig
        }
        

    def save_model(self, name="last", epoch=0):
        # save model
        print(f"saving model: {name} to {self.log_dir}")

        dict_config = {
            "model": self.accelerator.unwrap_model(self.model).state_dict(),
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }

        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))
        
