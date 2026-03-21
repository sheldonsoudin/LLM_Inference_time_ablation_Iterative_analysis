

"""
Sources: 
- nanoGPT  (Karpathy):   fused QKV, weight tying, Flash Attention, bias flag
- Raschka LLMs-from-Scratch: explicit custom LayerNorm, transparent GELU
- GPT-3 (Brown et al.):  alternating dense/sparse attention, 2048 ctx, scalekk

"""

import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from dataclasses import dataclass
from typing import Optional 

@dataclass
class GPTConfig: 
    #Modlel dimensions 
    n_layers: int = 128 #nbr of trasformer blocks  
    n_heads: int = 128  #nbr of attention heads
    n_embd: int = 16384  
    d_ffn: int = 65536 # x4 rule 
    context_len = 2048 #gpt-3's context window
    vocab_size: int = 50257 # bpt-2 byte paire encoder vocabulary 

    # Regularisation 
    dropout: float =0.1 

    #Archetecture flags
    bias: bool  = True # for bias i nLinear layers and layerNorm 
    flash: bool =  True #  

    sparse_block_size: int = 64 




class GPT(nn.module): 
    def forward(): 



class MLP(nn.module): 
    """
    Feed forward network
    Hidden dimension set in config

    """
    def __ini__(self,config: GPTConfig):
        




class LayerNorm(nn.Module): 
    """
    Using torch.nn.functional but still writting fct out transparently 
    """
    def __init__(self,ndim,bias:bool+True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None 
        self.epst = eps 

    def forward(self,x:torch.Tensor) -> torch.Tensor: 
        return F.layer_norm(x,self.weight.shape,self.weight, self.bias,self.eps) 



class Block(nn.module): 

    def __init__(self,config:GPTConfig,layer_idx:int): 
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd,bias=config.bias)
        self.attn = CausalSelfAttention(config,layer_idx) 
        self.ln_2 = LayerNorm(config.embd,bias=config.bias)
        self.mlp = MLP(config) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x+ self.attn(self.len_1(x)) # attention residual 
        x = x+ self.mlp(slef.ln_2(x)) # MLP residual 

