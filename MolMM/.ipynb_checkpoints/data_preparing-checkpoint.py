#!/usr/bin/env python
# coding: utf-8

from molmap import loadmap

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
#IPythonConsole.ipython_useSVG = True
import numpy as np
import pandas as pd
from tqdm import tqdm
from openbabel import pybel
import os
RDLogger.DisableLog('rdApp.*')


mp1 = loadmap('./test.mp')

import pickle
with open('flist.pkl', 'wb') as f:
    indices_list = mp1.flist
    pickle.dump(indices_list, f)

def clean_and_standardize(smiles,ph=7.4,iso=False):
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Skip invalid molecules
        if mol is None:
            return None,None

        # Canonicalize the SMILES
        # canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

        # Remove salts and other fragments / Keep only the largest fragment
        fragments = Chem.GetMolFrags(mol, asMols=True)
        largest_fragment = max(fragments, default=None, key=lambda m: m.GetNumAtoms())
        if largest_fragment is None:
            return None,None
        
        u = rdMolStandardize.Uncharger()
        uncharge_mol = u.uncharge(largest_fragment)
        uncharge_smiles = Chem.MolToSmiles(uncharge_mol, isomericSmiles=iso, canonical=True)
        
        ob_mol = pybel.readstring("smi", Chem.MolToSmiles(largest_fragment, isomericSmiles=iso, canonical=True))
        
        ob_mol.OBMol.AddHydrogens(False, True, ph)

        # Convert back to SMILES
        adjusted_smiles = ob_mol.write("smi").strip()

        return adjusted_smiles, uncharge_smiles
    
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None,None



def clean_csv(input_name,output_name):
    inh = pd.read_csv(input_name,encoding='utf-8')
    smiles_list = inh['smiles'].tolist()
    clean = []
    neu = []
    for smi in smiles_list:
        std_smi,neu_smi = clean_and_standardize(smi)
        clean.append(std_smi)
        neu.append(neu_smi)
    inh['smiles_pH'] = clean
    inh['smiles_neu'] = neu
    # inh = inh.dropna()
    inh.to_csv(output_name,index=False)


mp1 = loadmap('./test.mp')
def generate_MolDs(input_name,output_name,columns):
    df = pd.read_csv(input_name,encoding='utf-8')
    smi = df[columns]
    df_list = []
    for s in tqdm(smi):
        arr = mp1.extract.transform(s)
        df = pd.DataFrame(arr).T
        df.columns = mp1.extract.bitsinfo.IDs
        df_list.append(df)
    feat = pd.concat(df_list)
    feat.to_csv(output_name,index=False)
    
    rows_with_nan = np.array(feat.isna().any(axis=1),dtype=np.bool_)
    df = pd.read_csv(input_name,encoding='utf-8')
    df[columns+'_MolDs'] = rows_with_nan
    print(columns+'_MolDs')
    # assert False
    df.to_csv(input_name,index=False)
    
def scaling_min_max(df):
    df = mp1.MinMaxScaleClip(df,
            mp1.scale_info['min'], 
            mp1.scale_info['max'])
    df = df[mp1.flist]
    return df

def scaling_csv(name,name1):
    scaling_min_max(pd.read_csv(name)).to_csv(name1,index=False)


def generate_map(input_name,output_name,columns):
    df = pd.read_csv(input_name,encoding='utf-8')
    smi = df[columns].tolist()
    F = mp1.batch_transform(smi)
    np.savez(output_name,x=F)


def data_prepare(name,input_path,output_path):
    origin=os.path.join(input_path,name+'.csv')
    clean=os.path.join(input_path,name+'_clean.csv')
    
    if (os.path.exists(clean)):
        clean_csv(clean,clean)
    else:
        clean_csv(origin,clean)
    
    clean_MolDs=os.path.join(output_path,name+'_MolDs.csv')
    clean_minmax_MolDs=os.path.join(output_path,name+'_minmax_MolDs.csv')
    clean_npz=os.path.join(output_path,name+'.npz')
    
    if ('pH' in output_path):
        generate_MolDs(clean,clean_MolDs,'smiles_pH')
        scaling_csv(clean_MolDs,clean_minmax_MolDs)
        generate_map(clean,clean_npz,'smiles_pH')
    else:
        # assert False
        generate_MolDs(clean,clean_MolDs,'smiles_neu')
        scaling_csv(clean_MolDs,clean_minmax_MolDs)
        generate_map(clean,clean_npz,'smiles_neu')      


for name in ['inhibitors_refine']:
    data_prepare(name,'./csv/data_inh_refine/','./data/pH')
    data_prepare(name,'./csv/data_inh_refine/','./data/neu')


for name in ['substrates_refine']:
    data_prepare(name,'./csv/data_sub_refine/','./data/pH')
    data_prepare(name,'./csv/data_sub_refine/','./data/neu')


for name in ['competitive','excipients']:
    data_prepare(name,'./csv/data_extra/','./data/pH')
    data_prepare(name,'./csv/data_extra/','./data/neu')


for name in ['inhibitors_classes','substrates_classes']:
    data_prepare(name,'./csv/data_classes/','./data/pH')
    data_prepare(name,'./csv/data_classes/','./data/neu')


def sort_dict(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

def count_place(inh, parameter):
    count = {}
    for s in inh[parameter].unique():
        count[s] = sum(inh[parameter]==s)
    count = sort_dict(count)
    return count

inh=pd.read_csv('./csv/data_classes/inhibitors_classes.csv')
sub=pd.read_csv('./csv/data_classes/substrates_classes.csv')
inh_refine=pd.read_csv('./csv/data_inh_refine/inhibitors_refine.csv')
sub_refine=pd.read_csv('./csv/data_sub_refine/substrates_refine.csv')

from collections import defaultdict
smi=defaultdict(list)
for i in range(len(inh)):
    smi[inh.iloc[i]['smiles']].append(inh.iloc[i]['label'])
    
for i in range(len(sub)):
    smi[sub.iloc[i]['smiles']].append(sub.iloc[i]['label'])
    
val_list=[]
for k,v in smi.items():
    val_list.append(np.array(v).any())
    
df_merge=pd.DataFrame({'smiles':list(smi.keys()),'label':val_list})
df_merge.to_csv('./csv/data_extra/allocrites_classes.csv',index=False)


print(len(df_merge),count_place(df_merge,'label'))

smi=defaultdict(list)
for i in range(len(inh_refine)):
    smi[inh_refine.iloc[i]['smiles']].append(inh_refine.iloc[i]['label'])
    
for i in range(len(sub_refine)):
    smi[sub_refine.iloc[i]['smiles']].append(sub_refine.iloc[i]['label'])
    
val_list=[]
for k,v in smi.items():
    val_list.append(np.array(v).any())
    
df_merge=pd.DataFrame({'smiles':list(smi.keys()),'label':val_list})
df_merge.to_csv('./csv/data_extra/allocrites_refine.csv',index=False)

count=0
for k,v in smi.items():
    if len(v)==2:
        count+=1
print(count)

print(len(df_merge),count_place(df_merge,'label'))

for name in ['allocrites_classes','allocrites_refine']:
    data_prepare(name,'./csv/data_extra/','./data/pH')
    data_prepare(name,'./csv/data_extra/','./data/neu')





