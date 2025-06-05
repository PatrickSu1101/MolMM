import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.optim import AdamW
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, ConcatDataset
from pytorch_lightning.utilities import CombinedLoader
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import AUROC, Recall, Precision
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold
import os
from itertools import cycle

class DataModule(pl.LightningDataModule):
    def __init__(self, name=None, num_workers=1, batch_size=16, task_list=None, use_all=True, check_size=False):
        super().__init__()
        assert name is not None
        
        self.base_task_list=['inhibitors_classes','substrates_classes','tmp1','tmp2']
        # assert task_list is not None
        self.cwd=os.path.dirname(__file__)
        self.batch_size = batch_size
        self.name = name
        self.use_all = use_all
        self.num_workers=num_workers
        self.check_size=check_size
        
        self.task_list=task_list
        if task_list is not None:
            self.fit_task_list=[task for task in self.task_list if task not in self.base_task_list]
            if len(self.fit_task_list)==0:
                self.fit_task_list=[task for task in self.task_list if task in self.base_task_list]
            
        self.N_WAY=2
        
        self.mode="max_size_cycle"
        self.K_SHOT_SIZE = {
            'inhibitors_classes':4,
            'substrates_classes':4,
            'allocrites_classes':4,
            'inhibitors_refine':4,
            'substrates_refine':4,
            'allocrites_refine':4,
        }
            
        self.has_setup_fit = False
        self.has_prepare = False
        
        self.conf = {}
        self.smiles={}
        
        self.kfold_task = {
            'inhibitors_classes':5,
            'substrates_classes':5,
            'allocrites_classes':5,
            'inhibitors_refine':5,
            'substrates_refine':5,
            'allocrites_refine':5,
        }
        
        self.path = {
            'inhibitors_classes':'csv/data_classes/',
            'substrates_classes':'csv/data_classes/',
            'allocrites_classes':'csv/data_extra/',
            'inhibitors_refine':'csv/data_inh_refine/',
            'substrates_refine':'csv/data_sub_refine/',
            'allocrites_refine':'csv/data_extra/',
            'competitive':'csv/data_extra/',
            'excipients':'csv/data_extra/',
        }
        
        self.test_task = {
            'inhibitors_classes':'inhibitors_refine',
            'substrates_classes':'substrates_refine',
            'allocrites_classes':'allocrites_refine',
            'allocrites_refine':'competitive',
            'inhibitors_refine':'competitive',
            'substrates_refine':'competitive',
            'competitive':'competitive',
            'excipients':'excipients',
            # 'tmp1':'tmp1',
            # 'tmp2':'tmp2',
        }
    
    @staticmethod
    def split_dataset(dataset,ratio=[0.1,None],stratify=False,shuffle=True):
        target = dataset[:][1]
        s = target if (stratify) else None
        if ratio[1] is not None:
            train_valid_idx, test_idx= train_test_split(
                np.arange(len(target)),
                test_size=ratio[1],
                shuffle=True,
                stratify=s,
            )
        else:
            train_valid_idx = np.arange(len(target))
            test_idx = None
        
        s = target[train_valid_idx] if (stratify) else None
        ratio[1] = 0 if ratio[1] is None else ratio[1]
        
        train_idx, valid_idx= train_test_split(
            train_valid_idx,
            test_size=ratio[0]/(1-ratio[0]),
            shuffle=True,
            stratify=s,
        )
        
        if test_idx is not None:
            results = (Subset(dataset, train_idx),\
                       Subset(dataset, valid_idx),\
                       Subset(dataset, test_idx))
        else:
            results = (Subset(dataset, train_idx),\
                       Subset(dataset, valid_idx),\
                       None)
        
        return results
    
    @staticmethod
    def kfold(dataset,n_splits=5,kfold=True):
        train,test = [],[]
        if kfold:
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
            )
        else:
            skf = KFold(
                n_splits=n_splits,
                shuffle=True,
            )
        for train_idx,test_idx in skf.split(dataset[:][0],dataset[:][1]):
            train.append(Subset(dataset, train_idx))
            test.append(Subset(dataset, test_idx))
        return train,test
    
    @staticmethod
    def calc_iter(dataset_targets,K_shot):
        classes = torch.unique(dataset_targets).tolist()
        num_classes = len(classes)
        indices_per_class = {}
        batches_per_class = {}
        for c in classes:
            indices_per_class[c] = torch.where(dataset_targets == c)[0]
            batches_per_class[c] = indices_per_class[c].shape[0]//K_shot[int(c)]
            if batches_per_class[c] == 0:
                batches_per_class[c] += 1
        return min(batches_per_class.values())
        
    def exclude_data(self,task,refine_task):
        df_train = pd.read_csv(os.path.join(self.cwd,self.path[task],task+'_clean.csv'))
        df_test = pd.read_csv(os.path.join(self.cwd,self.path[refine_task],refine_task+'_clean.csv'))

        smiles_set = defaultdict(list)
        for smi in df_test['smiles']:
            smiles_set[smi].append(1)
        mask = np.array([ len(smiles_set[s])==0 for s in df_train['smiles'] ])
        
        try:
            smiles_set = defaultdict(list)
            for smi,v in zip(df_test['smiles'],df_test['label']):
                smiles_set[smi].append(v)
            mask = np.array([ len(smiles_set[s])==0 for s in df_train['smiles'] ])

            for smi,v in zip(df_train['smiles'],df_train['label']):
                smiles_set[smi].append(v)
            count=0
            conf_dict=defaultdict(list)
            for k,v in smiles_set.items():
                data_list = np.array(v)
                count += data_list.any() != data_list.all()
                if len(data_list)>1:
                    conf_dict[k]=data_list.any() != data_list.all()
                else:
                    conf_dict[k]=None
            # df_test['conflict'] = [ conf_dict[smi] for smi in df_test['smiles'] ]
            # df_test.to_csv(task+'_conflict.csv',index=False)
            self.conf[task] = count/(len(mask)-int(mask.sum()))
            print(task+'\'s overlap ',len(mask)-int(mask.sum()),(len(mask)-int(mask.sum()))/len(df_test))
            print(task+'\'s conflict ',count,count/(len(mask)-int(mask.sum())))
            print()
        except:
            print(task+'\'s overlap ',len(mask)-int(mask.sum()),(len(mask)-int(mask.sum()))/len(df_test))
            print(task+'\'s conflict ','No label for conflict!')
            print()
            
        return mask
    
    def exclude(self,task,task_):
        if task==task_:
            df=pd.read_csv(os.path.join(self.cwd,self.path[task],task+'_clean.csv'))
            return np.ones_like(len(df),dtype=np.bool_)
        print('excluding',task_,'from',task)
        mask1 = self.exclude_data(task,task_)
        if task_ == self.test_task[task_]:
            return mask1
        else:
            return mask1&self.exclude(task,self.test_task[task_])
        
    def prepare_data(self):
        if self.has_prepare:
            return None
        self.has_prepare = True
        task_list = self.test_task.keys()
        
        if 'CNN' in self.name:
            task_list = [ t for t in task_list if 'tmp' not in t ]

        feat = {}
        feat_pH,feat_neu,target={},{},{}
        if 'RAW' in self.name:
            with open('flist.pkl', 'rb') as f:
                flist = pickle.load(f)
            for task in task_list:
                name = task+'_MolDs.csv'
                feat_pH[task] = pd.read_csv(os.path.join(self.cwd,'data/pH/',name))[flist].to_numpy()
                feat_neu[task] = pd.read_csv(os.path.join(self.cwd,'data/neu/',name))[flist].to_numpy()
        elif 'DNN' in self.name:
            for task in task_list:
                if 'tmp' in task:
                    name = 'tmp_data.csv'
                    feat[task] = pd.read_csv(os.path.join(self.cwd,'data/tmp/',name)).iloc[:,:-5].to_numpy()
                    continue
                
                name = task+'_minmax_MolDs.csv'
                feat_pH[task] = pd.read_csv(os.path.join(self.cwd,'data/pH/',name)).to_numpy()
                feat_neu[task] = pd.read_csv(os.path.join(self.cwd,'data/neu/',name)).to_numpy()
        else:
            for task in task_list:
                name = task+'.npz'
                feat_pH[task] = np.load(os.path.join(self.cwd,'data/pH/',name))['x'].transpose((0,3,1,2))
                feat_neu[task] = np.load(os.path.join(self.cwd,'data/neu/',name))['x'].transpose((0,3,1,2))
        
        for task in task_list:
            # if 'tmp' in task:
            #     name = 'tmp_data.csv'
            #     if '1' in task:
            #         target[task] = pd.read_csv(os.path.join(self.cwd,'data/tmp/',name)).iloc[:,-3].to_numpy()
            #     else:
            #         target[task] = pd.read_csv(os.path.join(self.cwd,'data/tmp/',name)).iloc[:,-2].to_numpy()
                
            #     continue
            
            df = pd.read_csv(os.path.join(self.cwd,self.path[task],task+'_clean.csv'))
            mask = (~df['smiles_pH_MolDs']) & (~df['smiles_neu_MolDs'])

            if not self.check_size:
                if task == 'allocrites_classes':
                    mask = mask&self.exclude(task,self.test_task['substrates_classes'])
                    mask = mask&self.exclude(task,self.test_task['inhibitors_classes'])
                else:
                    mask = mask&self.exclude(task,self.test_task[task])
                
            self.smiles[task]=df['smiles'][mask].to_list()
            
            feat_pH[task] = feat_pH[task][mask]
            feat_neu[task] = feat_neu[task][mask]
            try:
                target[task] = df['label'][mask].to_numpy()
            except:
                target[task] = np.zeros(feat_pH[task].shape[0])
                
            feat[task] = feat_pH[task] if 'pH' in self.name else feat_neu[task]
        
        self.X,self.y = {},{}
        for task in task_list:
            self.X[task] = torch.tensor(feat[task],dtype=torch.float)
            self.y[task] = torch.tensor(target[task],dtype=torch.float)
            print(task, 'dataset_size:',len(self.X[task]))
            
        for task in task_list:
            if self.use_all and task in self.base_task_list:
                print('use allocrites_classes datasets',task)
                self.X[task] = torch.tensor(feat['allocrites_classes'],dtype=torch.float)
                self.y[task] = torch.tensor(target['allocrites_classes'],dtype=torch.float)
        print()
    
    def setup_task_v1(
        self,
        task,
        train_sub_data,
        train_data,
        val_sub_data,
        val_data=None,
        idx=None,
    ):
        is_no_val = 'nval' in self.name
        is_meta = 'meta' in self.name
        is_merge = 'merge' in self.name
        is_guide = 'guide' in self.name
        refine_task = self.test_task[task]
        fit_task = self.fit_task_list[0]
        
#        if train_data is None:
#            train_data=train_sub_data
#            val_data=val_sub_data
            
        if is_no_val:
            train_refine_x=torch.cat([train_data[:][0],val_data[:][0]])
            train_refine_labels=torch.cat([train_data[:][1],val_data[:][1]])
            val_refine_labels=train_refine_labels
            train_refine_data = TensorDataset(train_refine_x,train_refine_labels)
            val_refine_data=train_refine_data

        else:
            train_refine_x=train_data[:][0]
            train_refine_labels=train_data[:][1] 
            val_refine_x=val_data[:][0]
            val_refine_labels=val_data[:][1]
            train_refine_data = TensorDataset(train_refine_x,train_refine_labels)
            val_refine_data = TensorDataset(val_refine_x,val_refine_labels)
        
        if not is_no_val and (
            is_merge or
            refine_task in self.task_list
        ):
            train_merge_x = torch.cat([train_sub_data[:][0],val_sub_data[:][0]])
            train_merge_labels = torch.cat([train_sub_data[:][1],val_sub_data[:][1]])
            train_merge_data = TensorDataset(train_merge_x, train_merge_labels)
            val_merge_data = train_merge_data
            val_merge_labels = train_merge_labels
        else:
            train_merge_x = train_sub_data[:][0]
            train_merge_labels = train_sub_data[:][1]
            train_merge_data = TensorDataset(train_merge_x, train_merge_labels)
            val_merge_x = val_sub_data[:][0]
            val_merge_labels = val_sub_data[:][1]
            val_merge_data = TensorDataset(val_merge_x, val_merge_labels)

        if is_merge and task!=refine_task:
            if idx is not None:
                print('merge',idx,task,refine_task)
            else:
                print('merge',task,refine_task)
            train_merge_x = torch.cat([train_merge_x,train_refine_x],dim=0)
            train_merge_labels = torch.cat([train_merge_labels,train_refine_labels],dim=0)
            train_merge_data = TensorDataset(train_merge_x, train_merge_labels)
        else:
            if idx is not None:
                print('w/o merge',idx,task,refine_task)
            else:
                print('w/o merge',task,refine_task)
            
        if idx is not None:
            self.ft_dataset[idx][fit_task] = DataLoader(
                train_refine_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                #pin_memory=True,
            )
            if not is_no_val:
                self.ft_val_dataset[idx][fit_task] = DataLoader(
                    val_refine_data,
                    batch_size=len(val_refine_data),
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
        else:
            self.ft_dataset[fit_task] = DataLoader(
                train_refine_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                #pin_memory=True,
            )
            if not is_no_val:
                self.ft_val_dataset[fit_task] = DataLoader(
                    val_refine_data,
                    batch_size=len(val_refine_data),
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
        
        if is_meta and refine_task in self.task_list:
            pos_size = self.K_SHOT_SIZE[refine_task]
            neg_size = self.K_SHOT_SIZE[refine_task]
            K_shot_refine = np.array([neg_size,pos_size])
            batch_size=self.batch_size
            train_sampler_refine = TaskBatchSampler(
                train_refine_labels,
                N_way=self.N_WAY,
                K_shot=K_shot_refine,
                include_query=True,
                shuffle=True,
                batch_size=batch_size,
            )
            val_sampler_refine = TaskBatchSampler(
                val_refine_labels,
                N_way=self.N_WAY,
                K_shot=K_shot_refine,
                include_query=True,
                shuffle=True,
            )
        
        if is_meta:
            pos_size = self.K_SHOT_SIZE[task]
            neg_size = self.K_SHOT_SIZE[task]
            K_SHOT_train = np.array([neg_size,pos_size])
        
        batch_size=1
        if refine_task in self.task_list:
            if is_meta:
                print('multi_task',DataModule.calc_iter(train_merge_labels,K_SHOT_train),len(train_sampler_refine))
                batch_size=DataModule.calc_iter(train_merge_labels,K_SHOT_train)//len(train_sampler_refine)
            else:
                batch_size=len(train_merge_labels)//len(train_refine_labels)
        
        if is_meta:
            batch_size=self.batch_size*batch_size
            train_combine_data = train_merge_data
            train_sampler = TaskBatchSampler(
                train_merge_labels,
                self.N_WAY,
                K_SHOT_train,
                shuffle=True,
                include_query=True,
                batch_size=batch_size,
            )
            val_combine_data = val_merge_data
            val_sampler = TaskBatchSampler(
                val_merge_labels,
                self.N_WAY,
                K_SHOT_train,
                include_query=True,
                shuffle=True,
            )
            if 'guide' in self.name:
                # support: refine dataset; query-noisy: labeled dataset
                is_inv = 'inv' in self.name
                a_data =  train_refine_data if is_inv else train_merge_data
                a_labels = train_refine_labels if is_inv else train_merge_labels
                b_data =  train_merge_data if is_inv else train_refine_data
                b_labels = train_merge_labels if is_inv else train_refine_labels

                train_combine_data = ConcatDataset([a_data,b_data])
                train_sampler = TaskBatchSampler(
                    a_labels,
                    self.N_WAY,
                    K_SHOT_train,
                    b_labels,
                    shuffle=True,
                    batch_size=batch_size,
                )
                
                val_a_data = val_refine_data if is_inv else val_merge_data
                val_a_labels = val_refine_labels if is_inv else val_merge_labels
                val_b_data = val_merge_data if is_inv else val_refine_data
                val_b_labels = val_merge_labels if is_inv else val_refine_labels

                val_combine_data = ConcatDataset([a_data,b_data])
                val_sampler = TaskBatchSampler(
                    a_labels,
                    self.N_WAY,
                    K_SHOT_train,
                    b_labels,
                    shuffle=True,
                )
                
        if idx is not None:
            if is_meta:
                if refine_task in self.task_list:
                    self.train_dataset[idx][refine_task] = DataLoader(
                        train_refine_data,
                        batch_sampler=train_sampler_refine,
                        collate_fn=train_sampler_refine.get_collate_fn(),
                        num_workers=self.num_workers,
                        #pin_memory=True,

                    )
                    self.val_dataset[idx][refine_task] = DataLoader(
                        val_refine_data,
                        batch_sampler=val_sampler_refine,
                        collate_fn=val_sampler_refine.get_collate_fn(),
                        num_workers=self.num_workers,
                        #pin_memory=True,

                    )
                self.train_dataset[idx][task] = DataLoader(
                    train_combine_data,
                    batch_sampler=train_sampler,
                    collate_fn=train_sampler.get_collate_fn(),
                    num_workers=self.num_workers,
                    #pin_memory=True,

                )
                self.val_dataset[idx][task] = DataLoader(
                    val_combine_data,
                    batch_sampler=val_sampler,
                    collate_fn=val_sampler.get_collate_fn(),
                    num_workers=self.num_workers,
                    #pin_memory=True,

                )
            else:
                if refine_task in self.task_list:
                    self.train_dataset[idx][refine_task]=DataLoader(
                        train_refine_data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                    self.val_dataset[idx][refine_task]=DataLoader(
                        val_refine_data,
                        batch_size=len(val_refine_data),
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                self.train_dataset[idx][task]=DataLoader(
                    train_merge_data,
                    batch_size=int(self.batch_size*batch_size),
                    shuffle=True,
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
                self.val_dataset[idx][task]=DataLoader(
                    val_merge_data,
                    batch_size=len(val_merge_data),
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
        else:
            if is_meta:
                if refine_task in self.task_list:
                    self.train_dataset[refine_task] = DataLoader(
                        train_refine_data,
                        batch_sampler=train_sampler_refine,
                        collate_fn=train_sampler_refine.get_collate_fn(),
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                    self.val_dataset[refine_task] = DataLoader(
                        val_refine_data,
                        batch_sampler=val_sampler_refine,
                        collate_fn=val_sampler_refine.get_collate_fn(),
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                self.train_dataset[task] = DataLoader(
                    train_combine_data,
                    batch_sampler=train_sampler,
                    collate_fn=train_sampler.get_collate_fn(),
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
                self.val_dataset[task] = DataLoader(
                    val_combine_data,
                    batch_sampler=val_sampler,
                    collate_fn=val_sampler.get_collate_fn(),
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
            else:
                if refine_task in self.task_list:
                    self.train_dataset[refine_task]=DataLoader(
                        train_refine_data,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                    self.val_dataset[refine_task]=DataLoader(
                        val_refine_data,
                        batch_size=len(val_refine_data),
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                self.train_dataset[task]=DataLoader(
                    train_merge_data,
                    batch_size=int(self.batch_size*batch_size),
                    shuffle=True,
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
                self.val_dataset[task]=DataLoader(
                    val_merge_data,
                    batch_size=len(val_merge_data),
                    num_workers=self.num_workers,
                    #pin_memory=True,
                )
        
    def setup_task(self,task):
        
        refine_task = self.test_task[task]
        
        sub = TensorDataset(self.X[task], self.y[task])
        refine_sub = TensorDataset(self.X[refine_task], self.y[refine_task])
        
        scale = len(refine_sub)/len(sub)
        
        if task != refine_task:
            if 'kfold' in self.name:
                refine_path=os.path.join(self.cwd,'data/split/','_'.join(self.name[:4]),refine_task+'.pt')
                try:
                    train,val,test=torch.load(refine_path)
                    print('Loading split path,',refine_path)

                    ratio_refine = 1/((len(train[0])+len(val[0]))**0.25+1)
                    ratio_refine *= (1-1/self.kfold_task[task])
                    print(refine_task,'train_ratio',1-1/self.kfold_task[task]-ratio_refine)
                    print(refine_task,'val_ratio',ratio_refine)
                    print(refine_task,'test_ratio',1/self.kfold_task[task])
                    print()
                except:
                    try:
                        train_val,test = DataModule.kfold(refine_sub,self.kfold_task[task])
                    except:
                        train_val,test = DataModule.kfold(refine_sub,self.kfold_task[task],kfold=False)

                    ratio_refine = 1/(len(train_val[0])**0.25+1)
                    train,val = [],[]
                    for data in train_val:
                        try:
                            train_data,val_data,_=DataModule.split_dataset(data,ratio=[ratio_refine,None],stratify=True)
                        except:
                            train_data,val_data,_=DataModule.split_dataset(data,ratio=[ratio_refine,None],stratify=False)
                        train.append(train_data)
                        val.append(val_data)

                    ratio_refine *= (1-1/self.kfold_task[task])
                    print(refine_task,'train_ratio',1-1/self.kfold_task[task]-ratio_refine)
                    print(refine_task,'val_ratio',ratio_refine)
                    print(refine_task,'test_ratio',1/self.kfold_task[task])
                    print()
                    os.makedirs(os.path.join(self.cwd,'data/split/','_'.join(self.name[:4])),exist_ok=True)
                    torch.save([train,val,test],refine_path)
            else:
                ratio_refine = 1/(len(refine_sub)**0.25+1)
                try:
                    train,val,_=DataModule.split_dataset(refine_sub,ratio=[ratio_refine,None],stratify=True)
                except:
                    train,val,_=DataModule.split_dataset(refine_sub,ratio=[ratio_refine,None],stratify=False)
                print(refine_task,'train_ratio',1-ratio_refine)
                print(refine_task,'val_ratio',ratio_refine)
                print(refine_task,'test_ratio',0)
                print()
        else:
            print('No refine task,',task)
            train=[ None for _ in range(self.kfold_task[task]) ]
            val=[ None for _ in range(self.kfold_task[task]) ]
            test=[ None for _ in range(self.kfold_task[task]) ]

        if 'kfold' in self.name:
            path=os.path.join(self.cwd,'data/split/','_'.join(self.name[1:4]),task+'.pt')
                
            try:
                train_sub,val_sub,test_sub=torch.load(path)
                print('Loading split path,',path)
                
                ratio = 1/(len(train_val_sub[0])**0.25+1)
                ratio *= (1-1/self.kfold_task[task])
                print(task,'train_ratio',1-1/self.kfold_task[task]-ratio)
                print(task,'val_ratio',ratio)
                print(task,'test_ratio',1/self.kfold_task[task])
                print()
            except:
                try:
                    train_val_sub,test_sub = DataModule.kfold(sub,self.kfold_task[task])
                except:
                    train_val_sub,test_sub = DataModule.kfold(sub,self.kfold_task[task],kfold=False)
                
                ratio = 1/(len(train_val_sub[0])**0.25+1)
                train_sub,val_sub=[],[]
                for data in train_val_sub:
                    try:
                        train_data,val_data,_=DataModule.split_dataset(data,ratio=[ratio,None],stratify=True)
                    except:
                        train_data,val_data,_=DataModule.split_dataset(data,ratio=[ratio,None],stratify=False)
                    train_sub.append(train_data)
                    val_sub.append(val_data)
                ratio *= (1-1/self.kfold_task[task])
                print(task,'train_ratio',1-1/self.kfold_task[task]-ratio)
                print(task,'val_ratio',ratio)
                print(task,'test_ratio',1/self.kfold_task[task])
                print()
                os.makedirs(os.path.join(self.cwd,'data/split/','_'.join(self.name[1:4])),exist_ok=True)
                torch.save([train_sub,val_sub,test_sub],path)
        else:
            test_ratio = 0
            ratio=1/(len(sub)**0.25+1)
            try:
                train_sub,val_sub,_=DataModule.split_dataset(sub,ratio=[ratio,None],stratify=True)
            except:
                train_sub,val_sub,_=DataModule.split_dataset(sub,ratio=[ratio,None],stratify=False)
            print(task,'train_ratio',1-ratio-test_ratio)
            print(task,'val_ratio',ratio) 
            print(task,'test_ratio',test_ratio)
            print()

        if 'kfold' in self.name:
            for i,(train_sub_data,train_data,val_sub_data,val_data) in enumerate(zip(train_sub,train,val_sub,val)):
                self.setup_task_v1(task,train_sub_data,train_data,val_sub_data,val_data,idx=i)
        else:
            self.setup_task_v1(task,train_sub,train,val_sub,val,idx=None)
        print()
        
        if 'kfold' in self.name:
            for i,(test_data,ft_test_data) in enumerate(zip(test_sub,test)):
                if refine_task in self.task_list:
                    self.test_dataset[i][refine_task]=DataLoader(
                        test_data,
                        batch_size=len(test_data),
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                else:
                    self.test_dataset[i][task]=DataLoader(
                        test_data,
                        batch_size=len(test_data),
                        num_workers=self.num_workers,
                        #pin_memory=True,
                    )
                if ft_test_data is not None:
                    if refine_task in self.task_list:
                        self.ft_test_dataset[i][refine_task]=DataLoader(
                            ft_test_data,
                            batch_size=len(ft_test_data),
                            num_workers=self.num_workers,
                            #pin_memory=True,
                        )
                    else:
                        self.ft_test_dataset[i][task]=DataLoader(
                            ft_test_data,
                            batch_size=len(ft_test_data),
                            num_workers=self.num_workers,
                            #pin_memory=True,
                        )
                
    def setup(self, stage=None):
        if self.has_setup_fit:
            return None
        self.has_setup_fit = True
        dataset = {}
        for task in self.task_list:
            dataset[task] = TensorDataset(self.X[task], self.y[task])
    
        if ('kfold' in self.name):
            self.train_dataset=[ {} for _ in range(self.kfold_task[task]) ]
            self.val_dataset=[ {} for _ in range(self.kfold_task[task]) ]
            self.test_dataset = [ {} for _ in range(self.kfold_task[task]) ]
            self.ft_dataset = [ {} for _ in range(self.kfold_task[task]) ]
            self.ft_val_dataset = [ {} for _ in range(self.kfold_task[task]) ]
            self.ft_test_dataset = [ {} for _ in range(self.kfold_task[task]) ]
        else:
            self.train_dataset,self.val_dataset,self.test_dataset={},{},{}
            self.ft_dataset,self.ft_val_dataset,self.ft_test_dataset={},{},{}

        for task in self.task_list:
            assert task in ['inhibitors_classes','substrates_classes','substrates_refine','inhibitors_refine','tmp1','tmp2']
            if task in ['inhibitors_classes','substrates_classes','tmp1','tmp2']:
                self.setup_task(task)

    def train_dataloader(self):
        if ('kfold' in self.name):
            return [ CombinedLoader(data, mode=self.mode) for data in self.train_dataset ]
        else:
            return CombinedLoader(self.train_dataset, mode=self.mode)

    def val_dataloader(self):
        if ('kfold' in self.name):
            return [ CombinedLoader(data, mode=self.mode) for data in self.val_dataset ]
        else:
            return CombinedLoader(self.val_dataset, mode=self.mode)
        
    def test_dataloader(self):
        if ('kfold' in self.name):
            return [ CombinedLoader(data, mode=self.mode) for data in self.test_dataset ]
        else:
            return None
    
    def ft_dataloader(self):
        if ('kfold' in self.name):
            return [ CombinedLoader(data, mode=self.mode) for data in self.ft_dataset ]
        else:
            return CombinedLoader(self.ft_dataset, mode=self.mode)
        
    def ft_val_dataloader(self):
        if 'kfold' in self.name:
            if np.array([ len(data)!=0 for data in self.ft_val_dataset ]).all():
                return [ CombinedLoader(data, mode=self.mode) for data in self.ft_val_dataset ]
            else:
                return [ None for _ in range(5) ]
        else:
            if (len(self.ft_val_dataset)==0):
                return None
            return CombinedLoader(self.ft_val_dataset, mode=self.mode)
        
    def ft_test_dataloader(self):
        if ('kfold' in self.name):
            return [ CombinedLoader(data, mode=self.mode) for data in self.ft_test_dataset ]
        else:
            return None

class FewShotBatchSampler:
    def __init__(
        self,
        dataset_targets,
        N_way,
        K_shot,
        include_query=False,
        shuffle=False,
        shuffle_once=True,
        drop_last=False,
    ):
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_way = N_way
        self.K_shot = K_shot
        self.shuffle = shuffle
        self.include_query = include_query
        self.drop_last=drop_last
        if self.include_query:
            self.K_shot = K_shot*2
        self.batch_size = self.N_way * self.K_shot  # Number of overall images per batch
        # Organize examples by class
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide
        
        # Create a list of classes from which we select the N classes per batch
        for c in self.classes:
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0]//self.K_shot[int(c)]
            assert self.batches_per_class[c] > 0, 'K_shot too large!'
        
        self.iterations = min(self.batches_per_class.values())
            
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
        sort_idxs = [
            i + p * self.num_classes for i, c in enumerate(self.classes) for p in range(self.batches_per_class[c])
        ]
        self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()
        
        if (shuffle or shuffle_once):
            self.shuffle_data()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        
        # random.shuffle(self.class_list)
        
    def __len__(self):
        if self.include_query:
            return self.iterations*2
        return self.iterations

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()
        # Sample few-shot batches
        start_index = defaultdict(int)
        index_list=[]
        for it in range(self.iterations):
            class_batch = self.class_list[it * self.N_way : (it + 1) * self.N_way]  # Select N classes for the batchmm
            index_batch = []
            for c in class_batch:
                # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c]+self.K_shot[int(c)]])
                start_index[c] += self.K_shot[int(c)]
            if self.include_query:
                # index_batch = index_batch[::2] + index_batch[1::2]
                index_list.append(index_batch[::2] + index_batch[1::2])
                index_list.append(index_batch[1::2] + index_batch[::2])
            else:
                index_list.append(index_batch)
                
        for index_batch in index_list:
            yield index_batch

class TaskBatchSampler:

    def __init__(
        self,
        dataset_targets,
        N_way,
        K_shot,
        dataset_targets_1=None,
        batch_size=None,
        include_query=False,
        shuffle=False,
        shuffle_once=True,
        drop_last=False,
    ):
        super().__init__()
        N_way = int(max(dataset_targets)+1)
        self.batch_sampler = FewShotBatchSampler(
            dataset_targets,
            N_way,
            K_shot,
            include_query=include_query,
            shuffle=shuffle,
            shuffle_once=shuffle_once,
        )
        self.drop_last = drop_last
        
        self.local_batch_size = sum(self.batch_sampler.K_shot)
        self.total_iter = len(self.batch_sampler)
        
        if dataset_targets_1 is not None:
            self.batch_sampler_1 = FewShotBatchSampler(
                dataset_targets_1,
                N_way,
                K_shot,
                include_query=include_query,
                shuffle=shuffle,
                shuffle_once=shuffle_once,
            )
            self.local_batch_size = sum(self.batch_sampler.K_shot)+sum(self.batch_sampler_1.K_shot)
            self.total_iter = max(len(self.batch_sampler),len(self.batch_sampler_1))
            
        self.task_batch_size = batch_size if batch_size is not None else self.total_iter
        self.iter=self.total_iter//self.task_batch_size
        if (not self.drop_last and self.total_iter%self.task_batch_size!=0):
            self.iter+=1
        
        self.dataset_targets=dataset_targets
        self.dataset_targets_1=dataset_targets_1

    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        if self.dataset_targets_1 is not None:
            if len(self.batch_sampler) > len(self.batch_sampler_1):
                batch_iter=zip(self.batch_sampler,cycle(self.batch_sampler_1))
            else:
                batch_iter=zip(cycle(self.batch_sampler),self.batch_sampler_1)
            for batch_idx, (batch,batch_1) in enumerate(batch_iter):
                batch_1 = list(np.array(batch_1)+len(self.dataset_targets))
                batch_list.extend(batch+batch_1)
                if (batch_idx+1) % self.task_batch_size == 0:
                    yield batch_list
                    batch_list = []
            if not self.drop_last:
                if len(batch_list)>0:
                    yield batch_list
                    batch_list = []
        else:
            for batch_idx, batch in enumerate(self.batch_sampler):
                batch_list.extend(batch)
                if (batch_idx+1) % self.task_batch_size == 0:
                    yield batch_list
                    batch_list = []
            if not self.drop_last:
                if len(batch_list)>0:
                    yield batch_list
                    batch_list = []

    def __len__(self):
        return self.iter

    def get_collate_fn(self):
        # Returns a collate function that converts one big tensor into a list of task-specific tensors
        def collate_fn(item_list):
            imgs = torch.stack([img for img, target in item_list], dim=0)
            targets = torch.stack([target for img, target in item_list], dim=0)
            local_iter = targets.shape[0]//self.local_batch_size
            if targets.shape[0]%self.local_batch_size != 0:
                local_iter+=1
            
            if local_iter==self.task_batch_size:
                imgs = imgs.chunk(self.task_batch_size, dim=0)
                targets = targets.chunk(self.task_batch_size, dim=0)
            else:
                task_batch_size=self.total_iter%self.task_batch_size
                imgs = imgs.chunk(task_batch_size, dim=0)
                targets = targets.chunk(task_batch_size, dim=0)
            return list(zip(imgs,targets))
        return collate_fn
