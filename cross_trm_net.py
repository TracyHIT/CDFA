from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat
import torch
from torch import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math
from transformers import AutoModel,AutoConfig
from graph_trm import * 

from modules import *
from transformers import BertConfig, BertModel,BertLMHeadModel
from math import log
 



class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

 

        
class MLP(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(MLP, self).__init__()
        
        h = 1024
        self.layers = nn.Sequential(
            nn.Linear(input_size,2*h),
            nn.GELU(),
            nn.BatchNorm1d(2*h),
            nn.Linear(2*h, embedding_dim),
            nn.GELU(),  
            nn.BatchNorm1d(embedding_dim),
             
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
    
 
 

class CLDTrmDDSModelFinal(nn.Module):
    def __init__(self, device):
        super().__init__()

        hidden_dim = 384
        pool_kernel = 3

        self.mlp_fp = MLP(1024, hidden_dim) 
        self.mlp_fp2 = MLP(1024, hidden_dim) 
        self.mlp_fp3 = MLP(167, hidden_dim) 
        self.mlp_fp4 = MLP(1024, hidden_dim) 

        self.mlp_global = MLP(hidden_dim*3, hidden_dim) 
         
        self.pred_full1 =  nn.Sequential(
            # nn.BatchNorm1d(hidden_dim), 
            nn.Linear(hidden_dim,1024),
            nn.GELU(),   
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            # nn.Linear(1024,1024),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024,1024),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024,1024),
        ) 

        self.pred_full2 =  nn.Sequential(
            # nn.BatchNorm1d(hidden_dim*3), 
            nn.Linear(hidden_dim*3,1024),
            nn.GELU(),   
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            # nn.Linear(1024,1024),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024,1024),
            # nn.GELU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024,1024),
        ) 
        self.w_pred = nn.Sequential(
            nn.Linear(1024+1024,1 ),
            # nn.GELU(),
            # nn.Linear(1024,1)
        )
         
 
         
        self.gene_cnn = nn.Sequential(
            CDilated(1, hidden_dim,3),
            nn.GELU(),
            # GEGLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.MaxPool1d(pool_kernel),
            
            CDilated(hidden_dim, hidden_dim,3),
            nn.GELU(),
            # GEGLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.MaxPool1d(pool_kernel),
             
            CDilated(hidden_dim, hidden_dim,3),
            nn.GELU(),
            # GEGLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.MaxPool1d(pool_kernel),
            # eca_layer(hidden_dim)
        )

         
 
         
         
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dim_feedforward= 3072, activation='gelu', dropout=0.0 )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=0.0, dim_feedforward= 3072, activation='gelu',batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)    

          

    def forward(self,   fp1, fp2, fp12, fp22, fp13, fp23, fp14, fp24, cell_gene):
        # print(cell_gene.size())
        cell_gene = self.gene_cnn(cell_gene.unsqueeze(1)   )
        # cell_gene = self.gene_cnn(cell_gene   )
        cell_gene = Rearrange("b d n ->b n d")(cell_gene)
        
        drug1_features  = self.mlp_fp(fp1)
        drug2_features  = self.mlp_fp(fp2) 
        
        
         
        global_cell_feature = Reduce("b n d ->b d", reduction="mean")(  cell_gene  )
        global_features =  torch.stack([ global_cell_feature, drug1_features , drug2_features ], dim=1)
        global_features  = self.transformer_encoder(global_features)

        
        feature_cell2g = self.transformer_decoder( cell_gene,   global_features   )
        feature_cell2g =  Reduce("b n d ->b d", reduction="mean")( feature_cell2g  )
        global_features = Rearrange('b n d-> b (n d)')( global_features  )
         
        pred2  = self.pred_full2(  global_features  )
        pred1  = self.pred_full1(  feature_cell2g  )
        # pred = self.w_pred(   pred1 + pred2    )

        pred = self.w_pred(  torch.cat( (pred1, pred2), 1)     )
        return  pred # (pred1 + pred2 )/2


 