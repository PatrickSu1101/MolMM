import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
from argparse import ArgumentParser
from collections import defaultdict
import os
import re

from .modules_multi import *
from .dataset_multi import *

def get_models(args):
    name= args['name']
    task_list = args['task_list']
    result_dir = args['dir']
    ckpt_dir_1 = args['ckpt_dir_1']
    ckpt_dir_2 = ckpt_dir_1+'_fit'
    bar = args['bar']
    
    is_plot = len(ckpt_dir_1.split('/')[-1]) == 3
    if is_plot:
        print('Succeed to load plot model!')
    
    name_list = name.split('_')
    fit_task = task_list[0]
    best_model_path = None

    batch_size = 32
    num_epochs = 500
    num_epochs_fit = 200
    patience = 40
    patience_fit = 40
    
    folder_path = 'ckpt'
    ckpt_dict = defaultdict(str)
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            pathlist=os.path.join(dirpath, filename).split('/')
            if ('checkpoints' in dirpath and 'fit' not in dirpath and 'TMP' not in dirpath and len(pathlist[1])>3):
                ckpt_dict['/'.join([pathlist[1][-3:]]+pathlist[2:4])] = os.path.join(dirpath, filename)
    
    folder_path = 'ckpt'
    ckpt_dict_fit = defaultdict(str)
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            pathlist=os.path.join(dirpath, filename).split('/')
            if ('checkpoints' in dirpath and 'fit' in dirpath and 'TMP' not in dirpath and len(pathlist[1])>7):
                ckpt_dict_fit['/'.join([pathlist[1][-7:]]+pathlist[2:4])] = os.path.join(dirpath, filename)
    
    monitor,mode = 'val_total_acc','max'
    monitor_fit,mode_fit = 'val_total_acc','max'
    
    # if 'tmp' in name:
    #     monitor,mode = 'val_total_mae','min'
    #     monitor_fit,mode_fit = 'val_total_mae','min'
    
    N_way = {
        'inhibitors_classes':2,
        'substrates_classes':2,
        'allocrites_classes':2,
        'inhibitors_refine':2,
        'substrates_refine':2,
        'allocrites_refine':2,
        'guide':2,
        'competitive':0,
        'excipients':0,
        'tmp1':2,
        'tmp2':2,
    }
    N_way_fit = {
        'inhibitors_classes':2,
        'substrates_classes':2,
        'allocrites_classes':2,
        'inhibitors_refine':2,
        'substrates_refine':2,
        'allocrites_refine':2,
        'guide':2,
        'competitive':0,
        'excipients':0,
        'tmp1':2,
        'tmp2':2,
    }

    data_module=DataModule(name_list,num_workers=4,batch_size=batch_size,task_list=task_list)
    data_module.prepare_data()
    data_module.setup("fit")
    
    trainer_test=pl.Trainer(logger=CSVLogger(result_dir,name=name))
    
    if 'kfold' not in name_list:
        if is_plot:
            print('/'.join([ckpt_dir_2[-7:],name,'version_0']))
            path = ckpt_dict_fit['/'.join([ckpt_dir_2[-7:],name,'version_0'])]
            print('Loading checkpoing,',path)
            best_model_ = MultiTaskModel_fit.load_from_checkpoint(path)
            print('Success')
            return data_module.train_dataloader(),data_module.ft_dataloader(),None,best_model_
        
        model = MultiTaskModel(
            name=name_list,
            task_list=task_list,
            N_way=N_way,
            )
        model.init_model(model.model, data_module.train_dataloader())
        
        if ('pure' not in name):
            early_stopping = EarlyStopping(monitor=monitor,mode=mode,patience=patience,min_delta=1e-4)
            checkpoint_callback = ModelCheckpoint(monitor=monitor,mode=mode,save_top_k=1)
            trainer = pl.Trainer(
                strategy='ddp_find_unused_parameters_true',
                max_epochs=num_epochs,
                callbacks=[early_stopping,checkpoint_callback],
                logger=CSVLogger(ckpt_dir_1,name=name),
                enable_progress_bar=bar,
                enable_model_summary=False,
                use_distributed_sampler=False,
            ) 
            trainer.fit(model,data_module.train_dataloader(),data_module.val_dataloader())
            best_model_path=checkpoint_callback.best_model_path
            best_model=MultiTaskModel.load_from_checkpoint(best_model_path)
        else:
            best_model=model

        model = MultiTaskModel_fit(
            name=name_list,
            task_list=task_list,
            N_way=N_way_fit,
            has_val=has_val,
        )
        model.init_model(best_model.model,data_module.ft_dataloader())
        
        if 'stop' in name:
            best_model_=model
        else:
            ft_data,ft_val_data = data_module.ft_dataloader(), data_module.ft_val_dataloader()
            has_val = ft_val_data is not None
            
            early_stopping = EarlyStopping(monitor=monitor_fit,mode=mode_fit,patience=patience_fit,min_delta=1e-4)
            checkpoint_callback = ModelCheckpoint(monitor=monitor_fit,mode=mode_fit,save_top_k=1)
            trainer_ = pl.Trainer(
                strategy='ddp_find_unused_parameters_true',
                max_epochs=num_epochs_fit,
                callbacks=[early_stopping, checkpoint_callback] if has_val else [],
                logger=CSVLogger(ckpt_dir_2,name=name),
                enable_progress_bar=bar,
                enable_model_summary=False,
                use_distributed_sampler=False,
            )
            trainer_.fit(model,ft_data,ft_val_data)
            
            if not has_val:
                best_model_ = model
            else:
                best_model_path = checkpoint_callback.best_model_path
                best_model_ = MultiTaskModel_fit.load_from_checkpoint(best_model_path)
        
        return data_module.train_dataloader(),data_module.ft_dataloader(),None,best_model_
        
    for i, (train_data,val_data,test_data,ft_data,ft_val_data,ft_test_data) in enumerate(zip(
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
        data_module.ft_dataloader(),
        data_module.ft_val_dataloader(),
        data_module.ft_test_dataloader(),
    )):
        
        model = MultiTaskModel(
            name=name_list,
            task_list=task_list,
            N_way=N_way,
        )
        model.init_model(model.model,train_data)
        
        path = ckpt_dict['/'.join([ckpt_dir_1[-3:],'_'.join(name_list[:-1]+['stop']),'version_'+str(i)])]
        try:
            print('Loading checkpoing,',path)
            best_model=MultiTaskModel.load_from_checkpoint(path)
            print('Success')
        except:
            print('Failue and Begin to build PTMs')
            if 'pure' not in name:
                early_stopping = EarlyStopping(monitor=monitor,patience=patience,mode=mode,min_delta=1e-4)
                checkpoint_callback = ModelCheckpoint(monitor=monitor,mode=mode,save_top_k=1)
                trainer = pl.Trainer(
                    strategy='ddp_find_unused_parameters_true',
                    max_epochs=num_epochs,
                    callbacks=[early_stopping, checkpoint_callback],
                    logger=CSVLogger(ckpt_dir_1,name=name),
                    enable_progress_bar=bar,
                    enable_model_summary=False,
                    use_distributed_sampler=False,
                )

                trainer.fit(model,train_data,val_data)
                best_model_path=checkpoint_callback.best_model_path
                best_model=MultiTaskModel.load_from_checkpoint(best_model_path)
                
        if ('pure' in name):
            print('Training pure model!')
            best_model=model

        if 'stop' in name:
            print('Stop! because of its stop model')
            best_model_ = MultiTaskModel_fit.load_from_checkpoint(best_model_path)
            trainer_test.test(best_model_,test_data)
            trainer_test.test(best_model_,ft_test_data)
            continue

        has_val = ft_val_data is not None
        model = MultiTaskModel_fit(
            name=name_list,
            task_list=task_list,
            has_val=has_val,
            N_way=N_way_fit,
        )
        model.init_model(best_model.model,ft_data)

        early_stopping = EarlyStopping(monitor=monitor_fit,mode=mode_fit,patience=patience_fit,min_delta=1e-4)
        checkpoint_callback = ModelCheckpoint(monitor=monitor_fit,mode=mode_fit,save_top_k=1)
        trainer_ = pl.Trainer(
            strategy='ddp_find_unused_parameters_true',
            max_epochs=num_epochs_fit,
            callbacks=[] if not has_val else [early_stopping, checkpoint_callback],
            logger=CSVLogger(ckpt_dir_2,name=name),
            enable_progress_bar=bar,
            enable_model_summary=False,
            use_distributed_sampler=False,
        )
        trainer_.fit(model,ft_data,ft_val_data)

        if not has_val:
            best_model_ = model
        else:
            best_model_path = checkpoint_callback.best_model_path
            best_model_ = MultiTaskModel_fit.load_from_checkpoint(best_model_path)

        trainer_test.test(best_model_,test_data)
        trainer_test.test(best_model_,ft_test_data)
        
        if is_plot:
            return train_data,test_data,ft_test_data,best_model_

