

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
    n_heads = 128  #nbr of attention heads
    n_embd = 16384  
    d_ffn = 65536 # x4 rule 
    context_len = 2048 #gpt-3's context window
    vocab_size: 




class GPT(nn.module): 
    def forward(): 



class MLP(nn.module): 


class LayerNorm(): 



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

