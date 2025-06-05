import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW,SGD
from collections import OrderedDict, defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from pytorch_lightning.utilities import CombinedLoader
from torchmetrics import *
# from models_multi import DNN_single,LeNet_single,VAE
from torch.utils.data.dataset import ConcatDataset,Subset
import pandas as pd
import numpy as np
from copy import deepcopy
from statistics import mean, stdev
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

from .models_multi import *
            
class MultiTaskBase(pl.LightningModule):
    def __init__(self,task_list):
        self.save_hyperparameters()
        super(MultiTaskBase, self).__init__()
        
        base_task_list=['inhibitors_classes','substrates_classes','allocrites_classes']
        self.task_list=task_list
        self.fit_task_list=[task for task in self.task_list if task not in base_task_list]
        if len(self.fit_task_list)==0:
            self.fit_task_list=[task for task in self.task_list if task in base_task_list]
        
        self._reset_losses_dict()
    
    @staticmethod
    def solve_alpha(G, n_iter=50):
        """
        Frank-Wolfe algorithm.
        min_alpha (alpha^T G^T G alpha) s.t. alpha >= 0, sum(alpha) = 1
        """
        K = G.size(0)
        alpha = torch.ones(K, device=G.device) / K

        for t in range(n_iter):
            grad = 2 * torch.mm(G, alpha.unsqueeze(1)).squeeze()
            i = torch.argmin(grad)
            direction = torch.zeros_like(alpha)
            direction[i] = 1.0
            gamma = 2.0 / (t + 2.0)
            alpha = (1 - gamma) * alpha + gamma * direction

        alpha = alpha.clamp(min=0.0)
        alpha /= alpha.sum()

        return alpha
    
    @staticmethod
    def mgda_ub(gradients):
        eps = 1e-8
        num_step = 50
        with torch.no_grad():
            stacked_grads = torch.stack([torch.cat([g.view(-1) for g in gradients[task]]) for task in gradients.keys()])
            
            G = torch.mm(stacked_grads, stacked_grads.t())
            alpha = MultiTaskBase.solve_alpha(G,n_iter=50)
            
        return alpha
    
    @staticmethod
    def calculate_prototypes(features, targets):
        classes, _ = torch.unique(targets).sort()
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, classes
    
    def _reset_losses_dict(self):
        self.losses = {}
        for stage in ["train", "val", "test"]:
            self.losses[stage] = {}
            for task in self.task_list:
                self.losses[stage][task] = defaultdict(list)
            self.losses[stage]['total'] = defaultdict(list)

    def _get_mean_loss_dict_for_type(self, task):
        assert self.losses is not None
        mean_losses = {}
        for stage in ["train", "val", "test"]:
            for loss_fn_name in self.losses[stage][task].keys():
                mean_losses[stage + "_" + task + "_" + loss_fn_name] = torch.stack(
                    self.losses[stage][task][loss_fn_name]
                ).mean()
        return mean_losses
            
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            try:
                result_dict = {
                    "epoch": float(self.current_epoch),
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                }
            except:
                result_dict = {}
            for task in self.losses['val'].keys():
                result_dict.update(self._get_mean_loss_dict_for_type(task))
            self.log_dict(result_dict, sync_dist=True)
            
            sch = self.lr_schedulers()
            if isinstance(sch, ReduceLROnPlateau):
                sch.step(result_dict["train_total_loss"])
            
        self._reset_losses_dict()
        
    def on_test_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {}
            for task in self.losses['test'].keys():
                result_dict.update(self._get_mean_loss_dict_for_type(task))
            result_dict = {k: v for k, v in result_dict.items() if k.startswith("test")}
            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()
    
class MultiTaskModel(MultiTaskBase):
    def __init__(self,name, task_list, N_way):
        self.save_hyperparameters()
        super(MultiTaskModel, self).__init__(task_list)
        self.automatic_optimization = False
        self.name=name
        self.task_list=task_list
        self.N_way=N_way
        self.is_no_val='nval' in self.name
        
        self.weights=None
        self.num_inner_steps = 1 # 1-10
        self.lr_inner = 0.1
        self.lr = 1e-3
        self.decay_factor=0.5
        
        model_task_list=task_list
        if 'guide' in self.name:
            model_task_list=task_list+['guide']
        
        mode = self.name[0]
        if mode == 'CNN':
            self.model = LeNet_single(
                task_list=model_task_list,
                N_way=self.N_way
            )
        else:
            self.model = DNN_single(
                task_list=model_task_list,
                N_way=self.N_way
            )
            
        self.output_weight=nn.ParameterDict()
        self.output_bias=nn.ParameterDict()
        self.init_weight=None
        self.init_bias=None
        
        self.accuracy=nn.ModuleDict()
        self.auc=nn.ModuleDict()
        self.pr_auc=nn.ModuleDict()
        self.recall=nn.ModuleDict()
        self.precision=nn.ModuleDict()
        
        for task in self.task_list:
            self.accuracy[task]=Accuracy(task='multiclass',num_classes=N_way[task])
            self.pr_auc[task]=AveragePrecision(task='multiclass',num_classes=N_way[task])
            self.auc[task]=AUROC(task='multiclass',num_classes=N_way[task])
            self.recall[task]=Recall(task='multiclass',num_classes=N_way[task],average='macro')
            self.precision[task]=Precision(task='multiclass',num_classes=N_way[task],average='macro')
            
        self.losses = None
        self._reset_losses_dict()
    
    def run_model(self,local_model,task,x,is_meta=False,init=False,compute_loss=True):
        return local_model(x)[task], 0
#         if ('proto' in self.name):
#             if is_meta:
#                 if init:
#                     output_weight = self.init_weight
#                     output_bias = self.init_bias
#                 else:
#                     output_weight = self.output_weight[task]
#                     output_bias = self.output_bias[task]
#             else:
#                 output_weight = local_model.output_weight[task]
#                 output_bias = local_model.output_bias[task]

#             if ('vae' in self.name):
#                 results,_,vae_loss = local_model(x,self.task_list,compute_loss=compute_loss)
#                 feat_x=F.sigmoid(results[task])
#                 return F.linear(feat_x,output_weight,output_bias), vae_loss
            
#             feat_x=F.sigmoid(local_model(x,self.task_list)[task])
#             return F.linear(feat_x,output_weight,output_bias), 0
#         else:
#             if ('vae' in self.name):
#                 results,_,vae_loss = local_model(x,compute_loss=compute_loss)
#                 return results[task], vae_loss
#             return local_model(x)[task], 0
    
    def forward(self, task, x):
        return self.run_model(self.model,task,x)[0]
    
    def init_protos(self,local_model,task,support_x,support_l):
        if ('vae' in self.name):
            support_feats=F.sigmoid(local_model(support_x,self.task_list)[0][task])
        else:
            support_feats=F.sigmoid(local_model(support_x,self.task_list)[task])
        prototypes,_=MultiTaskModel.calculate_prototypes(support_feats,support_l)
        init_weight = 2*prototypes
        init_bias = -(torch.norm(prototypes, dim=1)**2)
        return init_weight,init_bias

    def init_model(self,model,training_data):
        self.weights={}
        for v,v_ in zip(self.model.parameters(),model.parameters()):
            if v.shape==v.shape:
                v.data=v_.data

        train = training_data.iterables
        for task in train.keys():
            dataset = train[task].dataset
            if isinstance(dataset,ConcatDataset):
                X = torch.concat([data[:][0] for data in dataset.datasets])
                y = torch.concat([data[:][1] for data in dataset.datasets])
            else:
                X,y = dataset[:][0],dataset[:][1]
            len_y=len(y)
            y_size,y_weight=[],[]
            for i in range(self.N_way[task]):
                y_size.append(int(sum(y==i)))
                y_weight.append(len_y/y_size[i])
            self.weights[task]=torch.tensor(np.array(y_weight),dtype=y.dtype)
            
            print(task+'_labels','total',[len_y],'size',y_size)

#            if ('proto' in self.name):
#                init_weight,init_bias = self.init_protos(self.model,task,X,y)
#                self.model.output_weight[task] = init_weight.detach()
#                self.model.output_bias[task] = init_bias.detach()
        print()
        
    def calc_loss(self,task,y1,query_l,weight=False):
        if not weight or self.weights is None:
            loss = F.cross_entropy(y1,query_l)
        else:
            self.weights[task]=self.weights[task].to(y1.device)
            loss = F.cross_entropy(y1,query_l,weight=self.weights[task])
        return loss

    def adapt_few_shot(self, task, batch, freeze=False, stage='train'):
        fit_task = self.fit_task_list[0]
        support_x,support_l=batch
        support_l=support_l.long()
            
        if (stage == 'fit'):
            local_model = self.model
            local_optim = self.optimizers()
        else:
            local_model = deepcopy(self.model)
            params = list(local_model.parameters())
            local_optim = SGD(params,lr=self.lr_inner)

        local_model.train()
        local_optim.zero_grad()
        for epoch in range(self.num_inner_steps):
            y1,loss_vae = self.run_model(local_model,fit_task,support_x)
            loss = self.calc_loss(task,y1,support_l,stage=='fit') + loss_vae
            
            self.manual_backward(loss)
            local_optim.step()
            local_optim.zero_grad()
        
        return local_model,loss.detach()
    
    def adapt_few_shot_proto(self, task, batch, freeze=False, stage='train'):
        fit_task=self.fit_task_list[0]
        support_x,support_l=batch
        support_l=support_l.long()
        
        if stage == 'fit':
            local_model = self.model
            local_optim = self.optimizers()
        else:
            local_model = deepcopy(self.model)
            self.init_weight,self.init_bias = self.init_protos(self.model,fit_task,support_x,support_l)
            local_model.output_weight[fit_task] = self.init_weight.detach()
            local_model.output_bias[fit_task] = self.init_bias.detach()
            params = list(local_model.parameters())
            local_optim = SGD(params,lr=self.lr_inner)
            
        local_model.train()
        local_optim.zero_grad()
        for epoch in range(self.num_inner_steps):
            y1,loss_vae = self.run_model(local_model,fit_task,support_x)
            loss = self.calc_loss(task,y1,support_l,stage=='fit') + loss_vae
            
            self.manual_backward(loss)
            local_optim.step()
            local_optim.zero_grad()
        
        if stage != 'fit':
            self.output_weight[fit_task]=(local_model.output_weight[fit_task]-self.init_weight).detach()+self.init_weight
            self.output_bias[fit_task]=(local_model.output_bias[fit_task]-self.init_bias).detach()+self.init_bias
            
        return local_model,loss.detach()
    
    def step(self, batch, stage):
        fit_task=self.fit_task_list[0]
        self.model.train()
        freeze = 'freeze' in self.name
        alpha = np.ones(len(self.task_list))/len(self.task_list)
        
        model_state = defaultdict(dict)
        for n,p in self.model.named_parameters():
            for task in self.task_list:
                model_state[n][task] = None
                
        loss_dict,acc_dict = {},{}
        for task in self.task_list:
            acc_dict[task]=0
            loss_dict[task]=0
            self.model.zero_grad()

            query_x,query_y=batch[task]
            query_l=query_y.long()

            local_model = self.model 

            y1,loss_vae = self.run_model(local_model,task,query_x)
            loss = self.calc_loss(task,y1,query_l,True) + loss_vae

            if stage in ['train']:
                self.manual_backward(loss)

            loss_dict[task]=loss.detach()
            self.losses[stage][task]['loss'].append(loss.detach())
            
            acc = self.accuracy[task](y1.detach(),query_l)
            self.losses[stage][task]['acc'].append(acc)
            acc_dict[task]=acc
                    
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    model_state[n][task] = p.grad

        if stage in ['train'] and len(self.task_list)>1:
            gradients = defaultdict(list)
            for n,p in self.model.named_parameters():
                if np.array([v is not None for v in model_state[n].values()]).all():
                    for task in model_state[n].keys():
                        gradients[task].append(model_state[n][task].detach())
            if not np.array( [len(v)==0 for v in gradients.values()] ).any():
                alpha = MultiTaskModel.mgda_ub(gradients)
            else:
                alpha = np.ones(len(gradients))/len(gradients)
  
        if stage in ['train']:
            self.model.zero_grad()
            for n,p in self.model.named_parameters():
                grad_list=[ v for v in model_state[n].values() if v is not None ]
                if len(grad_list)==len(model_state[n]):
                    p.grad=sum([ alpha[i]*grad_list[i] for i in range(len(alpha)) ])
                elif len(grad_list)>0:
                    p.grad=grad_list[0]
                    
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        if len(loss_dict.items())>1:
            if self.is_no_val:
                val_loss_list=[v for k,v in loss_dict.items() if k not in [fit_task]]
                val_acc_list=[v for k,v in acc_dict.items() if k not in [fit_task]]
            else:
                val_loss_list=[v for k,v in loss_dict.items()]
                val_acc_list=[v for k,v in acc_dict.items()]
            loss = sum(val_loss_list)/len(val_loss_list)
            total_acc = sum(val_acc_list)/len(val_acc_list)
        else:
            loss=loss_dict[fit_task]        
            total_acc=acc_dict[fit_task]

        self.losses[stage]['total']['loss'].append(loss)
        self.losses[stage]['total']['acc'].append(total_acc)

        return loss_dict,alpha

    def meta_step(self, batch, stage):
        fit_task = self.fit_task_list[0]
        freeze='freeze' in self.name
        alpha=np.ones(len(self.task_list))/len(self.task_list)
        is_guide='guide' in self.name and 'ab' not in self.name
        
        model_state = defaultdict(dict)
        for n,p in self.model.named_parameters():
            for task in self.task_list:
                model_state[n][task] = None
                
        loss_dict,acc_dict = {},{}
        for task in self.task_list:
            # initization of the metric and grad
            
            loss_dict[task]=0
            acc_dict[task]=0
            
            self.model.train()
            self.model.zero_grad()
            if is_guide:
                
                for n,p in self.model.named_parameters():
                    model_state[n]['guide'] = None
                
                loss_dict['guide']=0
                acc_dict['guide']=0
                guide_model=deepcopy(self.model)
                guide_model.train()
                guide_model.zero_grad()
                
            for i,task_batch in enumerate(batch[task]):

                s_x,q_x = task_batch[0].chunk(2,dim=0)
                s_y,q_y = task_batch[1].chunk(2,dim=0)
                s_batch,q_batch = (s_x,s_y),(q_x,q_y)
                if ('proto' in self.name):
                    local_model,_=self.adapt_few_shot_proto(task,s_batch,freeze=freeze)
                else:
                    local_model,_=self.adapt_few_shot(task,s_batch,freeze=freeze)
                
                query_x,query_y=q_batch
                query_l=query_y.long()
                
                y1,_ = self.run_model(local_model,fit_task,query_x)
                loss = self.calc_loss(task,y1,query_l,False)
                
                if stage in ['train']:
                    self.manual_backward(loss)
                    for p_global,p_local in zip(self.model.parameters(),local_model.parameters()):
                        if p_local.grad is not None:
                            if p_global.grad is None:
                                p_global.grad=p_local.grad
                            else:
                                p_global.grad+=p_local.grad
                                
                acc=self.accuracy[task](y1.detach(),query_l)
                #recall= self.recall[task](y1.detach().argmax(dim=1),query_l,)
                #precision=self.precision[task](y1.detach().argmax(dim=1),query_l)
                self.losses[stage][task]['loss'].append(loss.detach())
                self.losses[stage][task]['acc'].append(acc)
                #self.losses[stage][task]['recall'].append(recall)
                #self.losses[stage][task]['precision'].append(precision)
                loss_dict[task]+=loss.detach()
                acc_dict[task]+=acc
                
                # joint learning on support set
                if is_guide:
                    support_x,support_l=s_batch
                    support_l=support_l.long()
                    
                    y1,_ = self.run_model(guide_model,'guide',support_x)
                    loss = self.calc_loss(task,y1,support_l,False)
                    
                    if stage in ['train']:
                        self.manual_backward(loss)
                    
                    acc = self.accuracy[task](y1.detach(),support_l)
                    loss_dict['guide']+=loss.detach()
                    acc_dict['guide']+=acc
            
            # Save grad for calucalation of alpha
            for n,p in self.model.named_parameters():
                if p.grad is not None:
                    model_state[n][task]=p.grad
                    
            loss_dict[task]/=len(batch[task])
            acc_dict[task]/=len(batch[task])
            
            if is_guide:
                for n,p in guide_model.named_parameters():
                    if p.grad is not None:
                        model_state[n]['guide']=p.grad

                loss_dict['guide'] /= len(batch[task])
                acc_dict['guide'] /= len(batch[task])
        
        # calucalate alpha for JT
        if (stage in ['train'] and (len(self.task_list)>1 or is_guide)):
            gradients = defaultdict(list)
            for n,p in self.model.named_parameters():
                grad_list=[ v for v in model_state[n].values() if v is not None ]
                if len(grad_list)==len(model_state[n]):
                    for task in model_state[n].keys():
                        gradients[task].append(model_state[n][task].detach())
            if np.array([ len(v)!=0 for v in gradients.values() ]).all():
                alpha = MultiTaskModel.mgda_ub(gradients)
        
        # update the grad
        if stage in ['train']:
            self.model.zero_grad()
            for n,p in self.model.named_parameters():
                grad_list=[ v for v in model_state[n].values() if v is not None ]
                if len(grad_list)==len(model_state[n]):
                    p.grad=sum([ alpha[i]*grad_list[i] for i in range(len(alpha)) ])
                elif len(grad_list)>0:
                    p.grad=grad_list[0]
            
            opt=self.optimizers()
            opt.step()
            opt.zero_grad()
        
        if len(loss_dict.items())>1:
            if self.is_no_val:
                if 'inv' in self.name:
                    val_loss_list=[v for k,v in loss_dict.items() if k in [fit_task]]
                    val_acc_list=[v for k,v in acc_dict.items() if k in [fit_task]]
                else:
                    val_loss_list=[v for k,v in loss_dict.items() if k not in [fit_task]]
                    val_acc_list=[v for k,v in acc_dict.items() if k not in [fit_task]]
            else:
                val_loss_list=[v for k,v in loss_dict.items()]
                val_acc_list=[v for k,v in acc_dict.items()]
            loss = sum(val_loss_list)/len(val_loss_list)
            total_acc = sum(val_acc_list)/len(val_acc_list)
        else:
            loss=loss_dict[fit_task]        
            total_acc=acc_dict[fit_task]

        self.losses[stage]['total']['loss'].append(loss)
        self.losses[stage]['total']['acc'].append(total_acc)

        return loss_dict,alpha
    
    def training_step(self, batch, batch_idx):
        if ('meta' in self.name):
            self.meta_step(batch,'train')
        else:
            self.step(batch,stage='train')
    
    def validation_step(self, batch, batch_idx):
        if ('meta' in self.name):
            torch.set_grad_enabled(True)
            self.meta_step(batch,'val')
            torch.set_grad_enabled(False)
        else:
            self.step(batch,stage='val')
    
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.decay_factor,
            patience=10,
            min_lr=1e-6,
        )
        lr_scheduler = {
            "scheduler": scheduler,
        }
        return [optimizer], [lr_scheduler]

class MultiTaskModel_fit(MultiTaskModel):
    def __init__(
        self,
        name,
        task_list,
        N_way,
        has_val=False,
    ):
        super(MultiTaskModel_fit, self).__init__(name=name,task_list=task_list,N_way=N_way,)
        self.save_hyperparameters()
        self.num_inner_steps = 1
        self.has_val = has_val
        self._reset_losses_dict()
        
    def run_model(self,local_model,task,x):
        return local_model(x)[task], 0
        
    def step(self, batch, stage='val'):
        fit_task=self.fit_task_list[0]
        loss_dict,acc_dict,auc_dict,pr_auc_dict = {},{},{},{}
        for task in batch.keys():
            
            if task not in self.model.task_list and task not in self.weights:
                continue
            
            query_x,query_y=batch[task]
            query_y=query_y.long()

            y1,loss_vae = self.run_model(self.model,task,query_x)
            loss = self.calc_loss(task,y1,query_y,True)+loss_vae
            
            loss_dict[task]=loss.detach()
            acc_dict[task]=self.accuracy[task](y1.detach(),query_y)
            auc_dict[task]=self.auc[task](y1.detach(),query_y)
            pr_auc_dict[task]=self.pr_auc[task](y1.detach(),query_y)

            self.losses[stage][task]['loss'].append(loss_dict[task])
            self.losses[stage][task]['acc'].append(acc_dict[task])
            self.losses[stage][task]['auc'].append(auc_dict[task])
            self.losses[stage][task]['pr_auc'].append(pr_auc_dict[task])
            self.losses[stage][task]['recall'].append(self.recall[task](y1.detach(),query_y))
            self.losses[stage][task]['precision'].append(self.precision[task](y1.detach(),query_y))           
        self.losses[stage]['total']['loss'].append(loss_dict[fit_task])
        self.losses[stage]['total']['acc'].append(acc_dict[fit_task])
        self.losses[stage]['total']['auc'].append(auc_dict[fit_task])
        self.losses[stage]['total']['pr_auc'].append(pr_auc_dict[fit_task])
    
    def training_step(self,batch,batch_idx):
        freeze='freeze' in self.name
        fit_task=self.fit_task_list[0]
        fit_batch = batch[fit_task]
        if ('proto' in self.name):
            _,loss=self.adapt_few_shot_proto(fit_task,fit_batch,freeze=freeze,stage='fit')
        else:
            _,loss=self.adapt_few_shot(fit_task,fit_batch,freeze=freeze,stage='fit')
        self.losses['train']['total']['loss'].append(loss)
    
    def validation_step(self,batch,batch_idx):
        self.step(batch,stage='val')
    
    def test_step(self, batch, batch_idx):
        self.step(batch,stage='test')
    
    def configure_optimizers(self):
        if ('freeze' in self.name):
            params = list(self.model.lst.parameters()) + \
                     list(self.model.output_weight.parameters()) + \
                     list(self.model.output_bias.parameters())
        else:
            params = list(self.model.parameters())

        optimizer = AdamW(params,lr=self.lr)
            
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.decay_factor,
            patience=10,
            min_lr=1e-6,
        )
        lr_scheduler = {
            "scheduler": scheduler,
        }
        if self.has_val:
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer], []
        
        
#class MultiTaskModel_R(MultiTaskBase):
#    def __init__(
#        self,
#        name,
#        task_list,
#        N_way,
#    ):
#        self.save_hyperparameters()
#        super(MultiTaskModel_R, self).__init__(task_list)
#        self.automatic_optimization = False
#        
#        self.name = name
#        self.task_list = task_list
#        self.N_way=N_way
#        
#        self.num_inner_steps = 1
#        self.lr_inner = 0.1
#        self.lr = 1e-3
#        self.decay_factor=0.8
#        
#        mode = self.name[0]
#        if mode == 'CNN':
#            assert False
#        else:
#            self.model = DNN_single(240,512,task_list=task_list,N_way=N_way)
#        
#        self.MAE=nn.ParameterDict()
#        self.R2=nn.ParameterDict()
#        for task in self.task_list:
#            self.MAE[task] = MeanAbsoluteError()
#            self.R2[task] = R2Score()
#        
#        self.losses = None
#        self._reset_losses_dict()
#        
#    def run_model(self,local_model,task,x,is_meta=False,init=False,compute_loss=True):
#        return local_model(x)[task],0
#    
#    def forward(self,task,x):
#        return self.run_model(self.model,task,x)[0]
#    
#    def init_model(self,model,training_data):
#        for v,v_ in zip(self.model.parameters(),model.parameters()):
#            if (v.shape == v.shape):
#                v.data=v_.data
#
#        self.model.train()
#        train = training_data.iterables
#        for task in train.keys():
#            dataset = train[task].dataset
#            if isinstance(dataset,ConcatDataset):
#                X = torch.concat([data[:][0] for data in dataset.datasets])
#                y = torch.concat([data[:][1] for data in dataset.datasets])
#            else:
#                X,y = dataset[:][0],dataset[:][1]
#            print(task+'_labels','total',[len_y],[len(y)-y.sum(),y.sum()])
#        print()
#    
#    def calc_loss(self,task,y1,query_l,weight=False):
#        return F.mse_loss(y1,query_l)
#        # return F.cross_entropy(y1,query_l)
#    
#    def step(self, batch, stage):
#        self.model.train()
#        freeze = 'freeze' in self.name
#        alpha = np.ones(len(self.task_list))/len(self.task_list)
#        
#        model_state = defaultdict(dict)
#        for n,p in self.model.named_parameters():
#            for task in self.task_list:
#                model_state[n][task] = None
#                
#        loss_dict,mae_dict,r2_dict = {},{},{}
#        for task in self.task_list:
#            mae_dict[task]=0
#            loss_dict[task]=0
#            r2_dict[task]=0
#            self.model.zero_grad()
#
#            query_x,query_y=batch[task]
#            query_l=query_y
#
#            local_model = self.model 
#
#            y1,loss_vae = self.run_model(local_model,task,query_x)
#            y1 = y1.view(-1)
#            
#            loss = self.calc_loss(task,y1,query_l,True) + loss_vae
#
#            if stage in ['train']:
#                self.manual_backward(loss)
#
#            loss_dict[task]=loss.detach()
#            self.losses[stage][task]['loss'].append(loss.detach())
#            
#            mae = self.MAE[task](y1.detach(),query_l)
#            self.losses[stage][task]['mae'].append(mae)
#            mae_dict[task]=mae
#            
#            if stage in ['test']:
#                r2 = self.R2[task](y1.detach(),query_l)
#                self.losses[stage][task]['r2'].append(r2)
#                r2_dict[task]=r2
#                    
#            for n,p in self.model.named_parameters():
#                if p.grad is not None:
#                    model_state[n][task] = p.grad
#
#        if stage in ['train'] and len(self.task_list)>1:
#            gradients = defaultdict(list)
#            for n,p in self.model.named_parameters():
#                if np.array([v is not None for v in model_state[n].values()]).all():
#                    for task in model_state[n].keys():
#                        gradients[task].append(model_state[n][task].detach())
#            if not np.array( [len(v)==0 for v in gradients.values()] ).any():
#                alpha = MultiTaskModel.mgda_ub(gradients)
#            else:
#                alpha = np.ones(len(gradients))/len(gradients)
#  
#        if stage in ['train']:
#            self.model.zero_grad()
#            for n,p in self.model.named_parameters():
#                grad_list=[ v for v in model_state[n].values() if v is not None ]
#                if len(grad_list)==len(model_state[n]):
#                    p.grad=sum([ alpha[i]*grad_list[i] for i in range(len(alpha)) ])
#                elif len(grad_list)>0:
#                    p.grad=grad_list[0]
#            opt = self.optimizers()
#            opt.step()
#            opt.zero_grad()
#            
#        if stage in ['test']:
#            total_r2=sum(v for k,v in r2_dict.items())/len(self.task_list)
#            self.losses[stage]['total']['r2'].append(total_r2)
#        
#        loss=sum(v for k,v in loss_dict.items())/len(self.task_list)
#        total_mae=sum(v for k,v in mae_dict.items())/len(self.task_list)
#        self.losses[stage]['total']['loss'].append(loss)
#        self.losses[stage]['total']['mae'].append(total_mae)
#
#        return loss_dict,alpha
#    
#    def training_step(self, batch, batch_idx):
#        self.step(batch,stage='train')
#    
#    def validation_step(self, batch, batch_idx):
#        self.step(batch,stage='val')
#            
#    def test_step(self, batch, batch_idx):
#        self.step(batch,stage='test')
#    
#    def configure_optimizers(self):
#        optimizer = AdamW(self.model.parameters(),lr=self.lr)
#        scheduler = ReduceLROnPlateau(
#            optimizer,
#            "min",
#            factor=self.decay_factor,
#            patience=10,
#            min_lr=1e-6,
#        )
#        lr_scheduler = {
#            "scheduler": scheduler,
#        }
#        return [optimizer], [lr_scheduler]
#    
#class MultiTaskModel_R_fit(MultiTaskModel_R):
#    def __init__(
#        self,
#        name=None,
#        task_list=None,
#        N_way=None,
#    ):
#        self.save_hyperparameters()
#        super(MultiTaskModel_R_fit, self).__init__(name,task_list,N_way)
#        
