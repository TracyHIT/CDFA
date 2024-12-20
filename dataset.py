from torch.utils.data import Dataset
import requests
import numpy as np 
from tqdm import tqdm 
import torch 
import pandas as pd 

 
from rdkit import Chem
import dgl.backend as F
from rdkit.Chem import AllChem
from dgl import save_graphs, load_graphs
from joblib import Parallel, delayed, cpu_count
import json
from dgllife.utils import smiles_to_bigraph,WeaveAtomFeaturizer,CanonicalBondFeaturizer
from functools import partial
from torch.utils.data import DataLoader
import dgl.backend as F
import dgl
from transformers import  AutoTokenizer


from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_implicit_valence_one_hot, atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    atom_chiral_tag_one_hot, one_hot_encoding, bond_is_conjugated, atom_formal_charge, atom_num_radical_electrons, bond_is_in_ring, bond_stereo_one_hot
from dgllife.utils import BaseBondFeaturizer
from functools import partial




def init_featurizer():
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    # args = {
    #     'node_featurizer': MyAtomFeaturizer(),
    #     'edge_featurizer': MyBondFeaturizer()
    # }
    args = {
        'node_featurizer': WeaveAtomFeaturizer(atom_types = atom_types),
        'edge_featurizer': CanonicalBondFeaturizer(self_loop=True)
    }
     
    return args


class GraphFPDataset(Dataset):
    def __init__(self, smiles1, smiles2, celllines, labels, drug2fp, drug2fp2):
        self.labels =  labels 
        self.length = len(self.labels)
        self.smiles1 = smiles1
        self.smiles2 = smiles2 
        self.celllines = celllines
        self.drug2fp = drug2fp
        self.drug2fp2 = drug2fp2
        self.drug2graphs = {}
        config_drug_feature = init_featurizer()
        self._pre_process( smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                           node_featurizer=config_drug_feature['node_featurizer'],
                           edge_featurizer=config_drug_feature['edge_featurizer'],)
        

    def _pre_process(self, smiles_to_graph, node_featurizer,edge_featurizer):
        for  smile in tqdm( list(set(list(self.smiles1) + list(self.smiles2) ))):
            graph = smiles_to_graph(smile, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
            self.drug2graphs[smile] = graph
                
             

    def __getitem__(self, idx):
        
        graph_drug1 = self.drug2graphs[ str(self.smiles1[idx]) ]
        graph_drug2 = self.drug2graphs[ str(self.smiles2[idx]) ]
        
        label = self.labels[idx]
        
        cell_features = torch.FloatTensor(self.celllines[idx])
        drug1_fp1 = self.drug2fp[ self.smiles1[idx] ]
        drug2_fp1 = self.drug2fp[ self.smiles2[idx] ]

        drug1_fp2 = self.drug2fp2[ self.smiles1[idx] ]
        drug1_fp2 = self.drug2fp2[ self.smiles2[idx] ]


        return graph_drug1,graph_drug2, torch.FloatTensor(drug1_fp1), torch.FloatTensor(drug2_fp1),torch.FloatTensor(drug1_fp2), torch.FloatTensor(drug1_fp2), torch.FloatTensor(cell_features),  torch.FloatTensor([label]) 
        

    def __len__(self):
        return self.length

def collate_only_molgraphs(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    graph_drug1,graph_drug2, drug1_fp1, drug2_fp1, drug1_fp2, drug2_fp2, cell_features, labels = map(list, zip(*data))

    bg_drug1 = dgl.batch(graph_drug1)
    bg_drug1.set_n_initializer(dgl.init.zero_initializer)
    bg_drug1.set_e_initializer(dgl.init.zero_initializer)

    bg_drug2 = dgl.batch(graph_drug2)
    bg_drug2.set_n_initializer(dgl.init.zero_initializer)
    bg_drug2.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    cell_features = torch.stack(cell_features,dim=0) 
    drug1_fp1 = torch.stack(drug1_fp1,dim=0)  
    drug2_fp1 = torch.stack(drug2_fp1,dim=0)  

    drug1_fp2 = torch.stack(drug1_fp2,dim=0)  
    drug2_fp2 = torch.stack(drug2_fp2,dim=0)  


    return  bg_drug1  ,   bg_drug2 ,   drug1_fp1, drug2_fp1, drug1_fp2, drug2_fp2, cell_features, labels



class AllTrmDatasetV4(Dataset):
    def __init__(self,  smiles1, smiles2, celllines, labels, drug2fp, drug2fp2, drug2fp3, drug2fp4):
        self.labels =  labels 
        self.length = len(self.labels)
        self.smiles1 = smiles1 
        self.smiles2 = smiles2
        self.celllines = celllines
        self.drug2fp = drug2fp
        self.drug2fp2 = drug2fp2
        self.drug2fp3 = drug2fp3
        self.drug2fp4 = drug2fp4
         
     
    def __getitem__(self, idx):
        label = self.labels[idx]
        cell_features = torch.FloatTensor(self.celllines[idx])
        fp1 = torch.FloatTensor(self.drug2fp[ self.smiles1[idx] ])
        fp2 = torch.FloatTensor(self.drug2fp[ self.smiles2[idx] ])

        fp12 = torch.FloatTensor(self.drug2fp2[ self.smiles1[idx] ])
        fp22 = torch.FloatTensor(self.drug2fp2[ self.smiles2[idx] ])

        fp13 = torch.FloatTensor(self.drug2fp3[ self.smiles1[idx] ])
        fp23 = torch.FloatTensor(self.drug2fp3[ self.smiles2[idx] ])

        fp14 = torch.FloatTensor(self.drug2fp4[ self.smiles1[idx] ])
        fp24 = torch.FloatTensor(self.drug2fp4[ self.smiles2[idx] ])


        return   fp1, fp2, fp12, fp22, fp13, fp23, fp14, fp24, cell_features, torch.FloatTensor([label]) 

    def __len__(self):
        return self.length
 