

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


# model configuration
@dataclass
class GPTConfig: 
    #Modlel dimensions 
    n_layers: int = 24 #nbr of trasformer blocks  
    n_heads: int = 16  #nbr of attention heads
    n_embd: int = 1024  
    d_ffn: int = 4096 # x4 rule 
    context_len = 1048 #gpt-3's context window
    vocab_size: int = 50257 # bpt-2 byte paire encoder vocabulary 

    # Regularisation 
    dropout: float =0.1 

    #Archetecture flags
    bias: bool  = True # for bias i nLinear layers and layerNorm 
    flash: bool =  True #  

    sparse_block_size: int = 64 



# ####################### model components ##########
class GELU(nn.Module): 
    """
    same Gelu usde in gpt2/3 and nano gpt 
    written as explicit module following Raschka convention 
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return F.gelu(x)

class GPT(nn.module): 
    def forward(): 



class MLP(nn.module): 
    """
    Feed forward network
    Hidden dimension set in config

    """
    def __ini__(self,config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.d_ffn, bias = config.bias) 
        self.act = GELU()
        self.proj = nn.Linear(config.d_ffn, config.n_embd,bias=config.bias) 
        self.drop = nn.Dropout(config.dropout) 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x)))) 


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


class CausalSelfAttention(nn.Module): 

    """
    Multi head causal self-attention based on 

    nanoGPT : fused QKV projection, flash attention via F.scaled_dot_product_attention when use_flahs+True 

    gpt3: alternating dense / sparse attention cotrolled by 'layer_idx' , even layers (sull desne causal attention), odd layers (locally banded sparse attention) 

    """
    def __init__(self,config: GPTConfig, layer_idx: int): 
        super().__init__()
        assert config.n_embd % config.n_head == 0, \ 
        f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})" 

        self.head = config.n_head 
        self.head_dim = config.n_embd // config.n_head 
        self.n_embd = config.n_embd 
        self.dropout = config.dropout 
        self.use_flash = config.use_flash 
        self.sparse = ( layer_idx % 2 ==1) # odd layers are sparse 
        self.sparse_block = config.sparse_block_size 

        # fused QKV projection 
        self.qkv = nn.Linear(config.n_embd, 3*config.n_embd, bias = config.bias) 
        self.proj = nn.Linear(config.nn_embd, config.n_embd, bias = config.bias) 
        self.attn_drop = nn.Dropout(config.dropout) 
        self.resid_drop = nn.Dropout(config.dropout) 

        # causal 








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

