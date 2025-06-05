import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from collections import OrderedDict
import pandas as pd
import numpy as np
import pickle
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model,d_model)
        self.fc2 = nn.Linear(d_model,d_model)
        
    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))
    
class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class InceptionBlock_SiLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock_SiLU, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.reset_parameters()
        
    def forward(self, x):
        x1 = F.silu(self.branch1(x))
        x2 = F.silu(self.branch2(x))
        x3 = F.silu(self.branch3(x))
        return torch.cat([x1, x2, x3], dim=1)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.branch1.weight)
        nn.init.xavier_uniform_(self.branch2.weight)
        nn.init.xavier_uniform_(self.branch3.weight)
        
        self.branch1.bias.data.fill_(0)
        self.branch2.bias.data.fill_(0)
        self.branch3.bias.data.fill_(0)
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.reset_parameters()
        
    def forward(self, x):
        x1 = self.relu1(self.branch1(x))
        x2 = self.relu2(self.branch2(x))
        x3 = self.relu3(self.branch3(x))
        return torch.cat([x1, x2, x3], dim=1)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.branch1.weight)
        nn.init.xavier_uniform_(self.branch2.weight)
        nn.init.xavier_uniform_(self.branch3.weight)
        
        self.branch1.bias.data.fill_(0)
        self.branch2.bias.data.fill_(0)
        self.branch3.bias.data.fill_(0)
    
class LeNet_single(nn.Module):
    def __init__(self, task_list=[], N_way=None):
        super(LeNet_single, self).__init__()
        
        assert task_list is not None   
        self.task_list = task_list
        self.shared_layers = nn.Sequential(OrderedDict([
            ('conv',nn.Conv2d(13, 48, kernel_size=13, padding='same')),
            ('relu_conv',nn.ReLU()),
            ('pool_conv',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('incept1',InceptionBlock(48, 32)),
            ('pool_incept1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('incept2',InceptionBlock(96, 64)),
            ('pool_incept2',nn.AdaptiveMaxPool2d((1, 1))),
            ('flatten',nn.Flatten()),
            ('output',nn.Linear(64*3, 64)),
        ]))
        
        self.N_way=N_way
        self.output=nn.ModuleDict()
        self.lst=nn.ModuleDict()
        for task in self.task_list:
            self.output[task] = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            self.lst[task] = nn.Sequential(
                nn.Linear(32, self.N_way[task])
            )
        
        self.output_weight=nn.ParameterDict()
        self.output_bias=nn.ParameterDict()
        for task in self.task_list:
            self.output_weight[task]=nn.Parameter(torch.ones(self.N_way[task],32))
            self.output_bias[task]=nn.Parameter(torch.ones(self.N_way[task]))
        
        self.reset_parameters()
        
    def forward(self, x, ret_feat=[]):
        x = F.relu(self.shared_layers(x))
        results = {}
        for task in self.task_list:
            results[task] = self.output[task](x)
            if not (task in ret_feat):
                results[task] = self.lst[task](F.relu(results[task]))
            
        return results

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.shared_layers[0].weight)
        self.shared_layers[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.shared_layers[-1].weight)
        self.shared_layers[-1].bias.data.fill_(0)
        self.reset_output()
        self.reset_prototype()
    
    def reset_prototype(self):
        for task in self.task_list:
            self.output_weight[task].data.fill_(0)
            self.output_bias[task].data.fill_(0)
    
    def reset_output(self):
        for task in self.task_list:
            for i in range(2):
                nn.init.xavier_uniform_(self.output[task][i*2].weight)
                self.output[task][i*2].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lst[task][0].weight)
            self.lst[task][0].bias.data.fill_(0)
            
class LeNet_SHAP(LeNet_single):
    def __init__(self, task_list=[]):
        super(LeNet_SHAP, self).__init__(task_list=task_list)
        with open('indices_list.pkl', 'rb') as f:
            self.indices_list = pickle.load(f)
            
    def tab2image(self,x):
        arr_res = []
        for idict in self.indices_list:
            indices = idict['indices']
            idx = idict['idx']
            arr_1d = torch.zeros((x.shape[0],37*37),device=x.device)
            arr_1d[:,indices] = x[:,idx]
            arr_res.append(arr_1d.view(x.shape[0],1,37,37))
        x = torch.cat(arr_res, axis=1)
        return x
        
    def forward(self, x):
        x = self.tab2image(x)
        x = F.relu(self.shared_layers(x))
        task = self.task_list[0]
        return self.lst[task](F.relu(self.output[task](x)))
    
class DNN_single(nn.Module):
    def __init__(self,input_size=1344,hidden_size=128,task_list=None,N_way=None):
        super(DNN_single, self).__init__()
        self.task_list = task_list
        self.N_way=N_way
        self.shared_layers = nn.Sequential(OrderedDict([
            ('input', nn.Linear(input_size, hidden_size)),
            ('relu_input', nn.ReLU()),
            ('fc1', nn.Linear(hidden_size, hidden_size)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_size, hidden_size)),
        ]))
        
        self.output=nn.ModuleDict()
        self.lst=nn.ModuleDict()
        for task in self.task_list:
            self.output[task] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU(),
                nn.Linear(hidden_size//2, hidden_size//4),
            )
            self.lst[task] = nn.Sequential(
                nn.Linear(hidden_size//4, self.N_way[task])
            )
            
        self.output_weight=nn.ParameterDict()
        self.output_bias=nn.ParameterDict()
        for task in self.task_list:
            self.output_weight[task]=nn.Parameter(torch.ones(self.N_way[task],hidden_size//4))
            self.output_bias[task]=nn.Parameter(torch.ones(self.N_way[task]))
        
        self.reset_parameters()

    def forward(self, x, ret_feat=[]):
        x = F.relu(self.shared_layers(x))
        results = {}
        for task in self.task_list:
            results[task] = self.output[task](x)
            if not (task in ret_feat):
                results[task] = self.lst[task](F.relu(results[task]))
        return results
    
    def reset_parameters(self):
        for i in range(3):
            nn.init.xavier_uniform_(self.shared_layers[i*2].weight)
            self.shared_layers[i*2].bias.data.fill_(0)
        for task in self.task_list:
            for i in range(2):
                nn.init.xavier_uniform_(self.output[task][i*2].weight)
                self.output[task][i*2].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lst[task][0].weight)
            self.lst[task][0].bias.data.fill_(0)
    
class DNN_SHAP(DNN_single):
    def __init__(self, task_list=[]):
        super(DNN_SHAP, self).__init__(task_list=task_list)
        
    def forward(self, x):
        x = self.shared_layers(x)
        task = self.task_list[0]
        prob = self.lst[task](F.relu(self.output[task](x)))
        return prob
        
# class VAE(nn.Module):
#     """
#     Variational Autoencoder (VAE) class.
    
#     Args:
#         input_dim (int): Dimensionality of the input data.
#         hidden_dim (int): Dimensionality of the hidden layer.
#         latent_dim (int): Dimensionality of the latent space.
#     """
    
#     def __init__(self, task_list=None):
#         super(VAE, self).__init__()
        
#         assert task_list is not None   
#         self.task_list = task_list
        
#         self.latent_dim = 64
#         self.img_size = 37*37
#         self.shared_layers = nn.Sequential(OrderedDict([
#             ('conv',nn.Conv2d(13, 96, kernel_size=13, padding='same')),
#             ('relu_conv',nn.SiLU()),
#             ('pool_conv',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept1',InceptionBlock_SiLU(96, 64)),
#             ('pool_incept1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept2',InceptionBlock_SiLU(192, 128)),
#             ('pool_incept2',nn.AdaptiveMaxPool2d((1, 1))),
#             ('flatten',nn.Flatten()),
#             ('feedforward',nn.Linear(128*3, self.latent_dim*2)),
#         ]))
        
#         self.output=nn.ModuleDict()
#         self.lst=nn.ModuleDict()
#         for task in self.task_list:
#             self.output[task] = nn.Sequential(
#                 nn.Linear(self.latent_dim, self.latent_dim),
#                 nn.SiLU(),
#                 nn.Linear(self.latent_dim, self.latent_dim),
#                 nn.SiLU(),
#                 nn.Linear(self.latent_dim, 32),
#             )
#             self.lst[task] = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(32, 1)
#             )
            
#         self.output_weight=nn.ParameterDict()
#         self.output_bias=nn.ParameterDict()
#         for task in self.task_list:
#             self.output_weight[task]=nn.Parameter(torch.ones(2,32))
#             self.output_bias[task]=nn.Parameter(torch.ones(2))
        
#         self.softplus = nn.Softplus()
#         self.decoder = nn.Sequential(
#             nn.Linear(self.latent_dim, self.latent_dim*2),
#             nn.SiLU(),
#             nn.Linear(self.latent_dim*2, self.latent_dim*4),
#             nn.SiLU(),
#             nn.Linear(self.latent_dim*4, self.latent_dim*8),
#             nn.SiLU(),
#             nn.Linear(self.latent_dim*8, self.latent_dim*16),
#             nn.SiLU(),
#             nn.Linear(self.latent_dim*16, self.img_size),
#             nn.Sigmoid(),
#         )
#         self.reset_parameters()
        
#     def encode(self, x):
#         x = self.shared_layers(x)
#         mu, logvar = torch.chunk(x,2,dim=-1)
#         logvar = self.softplus(logvar) + 1e-8
#         return mu, logvar
        
#     def compute_kl_loss(self, mu, log_var):
#         return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return eps*std+mu
    
#     def decode(self, z):
#         return self.decoder(z)
    
#     def calc_tasks(self, z, ret_feat=[]):
#         results = {}
#         for task in self.task_list:
#             results[task] = self.output[task](z)
#             if not (task in ret_feat):
#                 results[task] = self.lst[task](results[task])
#             else:
#                 results[task] = results[task]
#         return results
    
#     def forward(self, x, ret_feat=[],compute_loss=True):
#         mu,logvar = self.encode(x)
#         if not compute_loss:
#             z = mu
#         else:
#             z = self.reparameterize(mu,logvar)
#         recon_x = self.decode(z)
#         results = self.calc_tasks(z,ret_feat)
        
#         if compute_loss:
#             kl_loss = self.compute_kl_loss(mu,logvar)
#             kl_loss = kl_loss.mean()
            
#             flat_x = x.sum(dim=1).view(z.size(0),-1)
#             recon_loss = F.mse_loss(recon_x,flat_x,reduction='none').sum(-1)
#             recon_loss = recon_loss.mean()
            
#             vae_loss = recon_loss + kl_loss
#             assert vae_loss < 10000
#         else:
#             vae_loss = 0
        
#         return results, z, vae_loss

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.shared_layers[0].weight)
#         self.shared_layers[0].bias.data.fill_(0)
#         self.reset_output()
#         self.reset_prototype()
#         for i in range(4):
#             nn.init.xavier_uniform_(self.decoder[i*2].weight)
#             self.decoder[i*2].bias.data.fill_(0)
    
#     def reset_prototype(self):
#         for task in self.task_list:
#             self.output_weight[task].data.fill_(0)
#             self.output_bias[task].data.fill_(0)
    
#     def reset_output(self):
#         for task in self.task_list:
#             for i in range(3):
#                 nn.init.xavier_uniform_(self.output[task][i*2].weight)
#                 self.output[task][i*2].bias.data.fill_(0)
#             nn.init.xavier_uniform_(self.lst[task][1].weight)
#             self.lst[task][1].bias.data.fill_(0)
    
# class LeNet_dual(nn.Module):
#     def __init__(self, ret_feat=False, task_list=None,input_channels=13, mode='dual'):
#         super(LeNet_dual, self).__init__()
        
#         assert task_list is not None
        
#         self.task_list = task_list
#         self.ret_feat = ret_feat
#         self.mode = mode
#         self.shared_layers1 = nn.Sequential(OrderedDict([
#             ('conv',nn.Conv2d(input_channels, 48, kernel_size=13, padding='same')),
#             ('silu_conv',nn.SiLU()),
#             ('pool_conv',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept1',InceptionBlock(48, 32)),
#             ('pool_incept1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept2',InceptionBlock(96, 64)),
#             ('pool_incept2',nn.AdaptiveMaxPool2d((1, 1))),
#             ('flatten',nn.Flatten()),
#         ]))
#         self.shared_layers2 = nn.Sequential(OrderedDict([
#             ('conv',nn.Conv2d(input_channels, 48, kernel_size=13, padding='same')),
#             ('silu_conv',nn.SiLU()),
#             ('pool_conv',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept1',InceptionBlock(48, 32)),
#             ('pool_incept1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept2',InceptionBlock(96, 64)),
#             ('pool_incept2',nn.AdaptiveMaxPool2d((1, 1))),
#             ('flatten',nn.Flatten()),
#         ]))
        
#         self.shared_layers = nn.Sequential(
#             self.shared_layers1,
#             self.shared_layers2,
#         )
        
#         if (self.mode == 'dual'):
#             self.input_dim = 64*3*2
#         else:
#             self.input_dim = 64*3
            
#         self.output=nn.ModuleDict()
        
#         for task in self.task_list:
#             self.output[task] = nn.Sequential(
#                 nn.Linear(self.input_dim, 128),
#                 nn.SiLU(),
#                 nn.Linear(128, 32),
#                 nn.SiLU(),
#                 nn.Linear(32, 16),
#             )
#             if (self.ret_feat):
#                 self.output[task].append(nn.Sigmoid())
#             else:
#                 self.output[task].append(nn.SiLU())
#                 self.output[task].append(nn.Linear(16, 1))
            
#         self.reset_parameters()
        
#     def forward(self, x):
#         if (self.mode in ['dual']):
#             inputx = x.chunk(2,dim=1)
#             x0 = self.shared_layers[0](inputx[0])
#             x1 = self.shared_layers[1](inputx[1])
#             x = torch.cat([x0,x1],dim=1)
#         else:
#             x = self.shared_layers[0](x) + self.shared_layers[1](x).sum() * 0
        
#         if (self.ret_feat):
#             x = F.sigmoid(x)
#             return x
        
#         results = {}
#         for task in self.task_list:
#             results[task] = self.output[task](x)
        
#         return results

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.shared_layers[0][0].weight)
#         self.shared_layers[0][0].bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.shared_layers[1][0].weight)
#         self.shared_layers[1][0].bias.data.fill_(0)
#         for task in self.task_list:
#             for i in range(3):
#                 nn.init.xavier_uniform_(self.output[task][i*2].weight)
#                 self.output[task][i*2].bias.data.fill_(0)
        
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
        
#         # Shared layers
#         self.shared_layers = nn.Sequential(OrderedDict([
#             ('conv1',nn.Conv2d(13, 48, kernel_size=13, padding='same')),
#             ('pool1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('silu1',nn.SiLU()),
#         ]))
        
#         self.task1_layers = nn.Sequential(OrderedDict([
#             ('incept1',InceptionBlock(48, 32)),
#             ('pool1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept2',InceptionBlock(96, 64)),
#             ('global_pool',nn.AdaptiveMaxPool2d((1, 1))),
#             ('flatten',nn.Flatten()),
#             ('fc1',nn.Linear(64*3, 128)),
#             ('silu1',nn.SiLU()),
#             ('fc2',nn.Linear(128, 32)),
#             ('silu2', nn.SiLU()),
#             ('output', nn.Linear(32, 1)),
#         ]))
        
#         self.task2_layers = nn.Sequential(OrderedDict([
#             ('incept1',InceptionBlock(48, 32)),
#             ('pool1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ('incept2',InceptionBlock(96, 64)),
#             ('global_pool',nn.AdaptiveMaxPool2d((1, 1))),
#             ('flatten',nn.Flatten()),
#             ('fc1',nn.Linear(64*3, 128)),
#             ('silu1',nn.SiLU()),
#             ('fc2',nn.Linear(128, 32)),
#             ('silu2', nn.SiLU()),
#             ('output', nn.Linear(32, 1)),
#         ]))
        
#     def forward(self, x, task=None):
#         # Shared layers
#         x = self.shared_layers(x)
        
#         if (task is not None):
#             return self.task1_layers(x) if (task == 1) else self.task2_layers(x)
#         # Task-specific layers
#         task1_output = F.sigmoid(self.task1_layers(x))
#         task2_output = F.sigmoid(self.task2_layers(x))
        
#         return task1_output, task2_output


# class DNN(nn.Module):
#     def __init__(self,input_size=629,hidden_size=512):
#         super(DNN, self).__init__()
        
#         self.shared_layers = nn.Sequential(OrderedDict([
#             ('input', nn.Linear(input_size, hidden_size)),
#             ('silu_input', nn.SiLU()),
#             ('fc1', nn.Linear(hidden_size, hidden_size)),
#             ('silu1', nn.SiLU()),
#         ]))
        
#         self.task1_layers = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(hidden_size, hidden_size)),
#             ('silu1', nn.SiLU()),
#             ('fc2', nn.Linear(hidden_size, hidden_size)),
#             ('silu2', nn.SiLU()),
#             # ('fc3', nn.Linear(hidden_size, hidden_size)),
#             # ('silu3', nn.SiLU()),
#             # ('fc4', nn.Linear(hidden_size, hidden_size)),
#             # ('silu4', nn.SiLU()),
            
#             ('output', nn.Linear(hidden_size, 1)),
#         ]))
        
#         self.task2_layers = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(hidden_size, hidden_size)),
#             ('silu1', nn.SiLU()),
#             ('fc2', nn.Linear(hidden_size, hidden_size)),
#             ('silu2', nn.SiLU()),
#             # ('fc3', nn.Linear(hidden_size, hidden_size)),
#             # ('silu3', nn.SiLU()),
#             # ('fc4', nn.Linear(hidden_size, hidden_size)),
#             # ('silu4', nn.SiLU()),
            
#             ('output', nn.Linear(hidden_size, 1)),
#         ]))

#     def forward(self, x, task=None):
#         # Shared layers
#         x = self.shared_layers(x)
        
#         if (task is not None):
#             return self.task1_layers(x) if (task == 1) else self.task2_layers(x)
#         # Task-specific layers
#         task1_output = F.sigmoid(self.task1_layers(x))
#         task2_output = F.sigmoid(self.task2_layers(x))
        
#         return task1_output, task2_output


# class Proto_C(nn.Module):
#     def __init__(
#         self
#     ):
#         super(Proto_C, self).__init__()
#         self.feat_size = 64
#         self.num_prototypes = 1
#         self.create_prototypes()
        
#     def create_prototypes(self) -> None:
#         self.prototype_vectors = nn.Parameter(torch.rand(self.num_prototypes, self.feat_size), requires_grad=True)
    
#     def calc_dist(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.cdist(x, self.prototype_vectors, p=2)
        
#     def forward(self, feat, stage='train'):
#         dist = self.calc_dist(feat)
#         return dist
    
    
# class PPN(nn.Module):
#     def __init__(
#         self,
#         model,
#     ):
#         super(PPN, self).__init__()

#         self.feat_size = 16
#         self.proto_range = (0,15)
#         self.num_prototypes = (self.proto_range[1] - self.proto_range[0]) + 1
#         self.interval = self.proto_range[1]-self.proto_range[0]
#         self.model_feat = model

#         # self.emb_loss_s = nn.Parameter(torch.ones((self.num_prototypes))*10, requires_grad=True)
#         self.emb_loss_s = nn.Parameter(torch.ones((1))*10, requires_grad=True)
#         self.sigma = nn.Parameter(torch.tensor([1.0]),requires_grad=False)
#         self.w_leak = nn.Parameter(torch.tensor([0.05]), requires_grad=False)
#         self.r = nn.Parameter(torch.tensor([5.0]), requires_grad=False)
#         # self.r = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        
#         self.create_prototypes()
        
#     def create_prototypes(self) -> None:
#         # Create prototypes and prototype classes
#         self.prototype_vectors = nn.Parameter(torch.rand(self.num_prototypes, self.feat_size), requires_grad=True)
#         proto_classes = torch.linspace(self.proto_range[0],self.proto_range[1],int(self.num_prototypes))

#         # Save all proto info
#         self.register_buffer("proto_classes", proto_classes.float())
#         self.register_buffer(
#             "prototype_feat",
#             torch.zeros(self.num_prototypes, self.feat_size, dtype=torch.uint8),
#         )
        
#     def forward(self, x, y=None, stage='train') -> dict:
#         x = self.model_feat(x)
#         dist = self.calc_dist(x)
#         info_dict = {
#             "x": x,
#             "dist": dist,
#             "proto_classes":self.proto_classes,
#         }
#         if (y is not None):
#             info_dict.update(self.calc_loss(dist, y))
#         info_dict.update(self.predict_forward(dist))
#         return info_dict
    
#     def calc_dist(self, x: torch.Tensor) -> torch.Tensor:
#         wdist = torch.cdist(x, self.prototype_vectors, p=2) * self.emb_loss_s
#         return wdist
    
#     def calc_loss(self, dist: torch.Tensor, label: torch.Tensor) -> dict:
#         label_samples = label.unsqueeze(1)
#         label_prototypes = self.proto_classes.unsqueeze(0)
#         l_dist = torch.abs(label_samples-label_prototypes)
        
#         beta = ((self.sigma)**-2)/2
#         w = (torch.exp(-beta * l_dist**2) + self.w_leak).detach()
#         w_diff = w * torch.abs(dist - l_dist)
        
#         loss = w_diff.sum() / (w.sum() + 1e-9)
        
#         return {'loss': loss}
    
#     def predict_forward(self, min_distances: torch.Tensor) -> dict:
#         # cut of distances at self.r and generate gaussian weights
#         clipped_dists = torch.where(min_distances > self.r, torch.inf, min_distances)
#         # clipped_dists = min_distances
#         beta = ((self.r.to(min_distances.device)/3)**-2)/2
#         w = torch.exp(-beta * clipped_dists**2)
        
#         # make prediction
#         class_IDx = self.proto_classes.unsqueeze(0)
#         prediction_raw = torch.sum(class_IDx * w, dim=1) / torch.sum(w, dim=1)

#         # check if prediction contains nans
#         mask_na = torch.isnan(prediction_raw)
#         numnans = mask_na.sum()

#         if numnans > 0:
#             # get closest prototype to each sample
#             sum_w = torch.sum(w, dim=1)
#             closest_proto_index = torch.argmin(min_distances, dim=1)
#             closest_proto_label = self.proto_classes[closest_proto_index]  # type:ignore

#             # if the weights of the prototypes are all 0, use the label of the closest prototype
#             prediction = torch.where(mask_na, closest_proto_label, prediction_raw)

#             # update the weights accordingly (used for proto matching matrix)
#             for i in range(sum_w.shape[0]):
#                 if mask_na[i]:
#                     w[i, closest_proto_index[i]] = 1
#         else:
#             prediction = prediction_raw

#         return {
#             "y": prediction,
#             "mask": mask_na,
#             "weights": w,
#         }
