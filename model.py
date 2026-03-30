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
    n_layer: int = 24 #nbr of trasformer blocks  
    n_head: int = 16  #nbr of attention heads
    n_embd: int = 1024  
    d_ffn: int = 4096 # x4 rule 
    n_ctx: int = 1024 #gpt-3's context window
    vocab_size: int = 50257 # bpt-2 byte paire encoder vocabulary 

    # Regularisation 
    dropout: float =0.1 

    #Archetecture flags
    bias: bool  = True # for bias i nLinear layers and layerNorm 
    flash: bool =  True #  
    sparse_block_size: int = 64 

    def __post_init__(self):
            assert self.n_embd % self.n_head == 0, (
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head}); "
                f"head_dim = {self.n_embd // self.n_head}"
            )
            assert self.d_ffn == 4 * self.n_embd, (
                f"d_ffn ({self.d_ffn}) should be 4 × n_embd ({self.n_embd}) = {4*self.n_embd}. "
                f"Override intentionally if you want a different ratio."
            )



# ####################### model components ##########
class GELU(nn.Module): 
    """
    same Gelu usde in gpt2/3 and nano gpt 
    written as explicit module following Raschka convention 
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return F.gelu(x)



class GPT(nn.Module): 
    """"
    Embedding and output head 
    - nanoGPT/GPT-3 : token embeddin wte and positional embedding wpe 
    - nanoGPT: lm_head shares weights with wte 
    - n_embd = 1024 
    - nanoGPT : lm_head has no bias 

    Initialisaztion: scaled initialisation from nanoGPT 
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict( 
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.n_ctx, config.n_embd),
                drop = nn.Dropout(config.dropout), 
                h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias = config.bias), 
        ))

        # Lm head 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size,bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        # initialisation 
        self.apply(self._init_weights)

        # scaling 
        proj_scale = (2*config.n_layer)**-0.5
        for name, p in self.named_parameters(): 
            if name.endswith("proj.weight") or name.endswith("qkv.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02*proj_scale)
        print(f" gpt initialisation done - {self.num_params():,} number of parameters"
              f"{self.num_params()/1e6:.1f}M)") 


    # helpers 
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int: 
        """ total trainable parameter count """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(
        self, 
        lr: float, 
        weight_decay: float, 
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda", 
    ) -> torch.optim.AdamW: 
        """ 
        AdamW with weight decay on weights only 
        Biases and LayerNorm parameters excluded from decay 
        returns a fused adamW aon cuda
        """
        decay, no_decay = set(), set() 
        for name, p in self.named_parameters(): 
            if not p.requires_grad: 
                continue 
            if p.dim() <2 or name.endswith(".bias"):
                no_decay.add(name)
            else:
                decay.add(name)

        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        param_groups = [ 
            { "params": [param_dict[n] for n in sorted(decay)], "weight_decay": weight_decay}, 
            { "params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0}, 
        ]

        use_fused = (device_type == "cuda") 
        extra = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, **extra)


    ##foward pass 
    def forward( 
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None, 
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        returns (logits,loss)
        inference mode: loss is none when targest is none 
        """

        B,T = idx.shape 
        assert T <= self.config.n_ctx, "sequence length exceeds max context" # TODO specific 

        device = idx.device 
        pos = torch.arange(T,device=device).unsqueeze(0) # (1,T)

        #embededing 
        tok_emb = self.transformer.wte(idx) #(B,T,n_embd)
        pos_emb = self.transformer.wpe(pos) #(1,T,n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        #transformer blocks 
        for block in self.transformer.h: 
            x = block(x)

        # final norm and project vocab 
        x = self.transformer.ln_f(x) # (B,T,n_embd)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # loss 
        
        loss = None 
        if targets is not None: 
            loss = F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1),
                ignore_index=-1, 
            )
        return {"logits": logits, "loss": loss}

    ## inference helper 
    @torch.no_grad() 
    def generate(
        self,
        idx: torch.Tensor, # (B,T) , prompt ids 
        max_new: int = 128, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None,
    ) -> torch.Tensor : 
        """ 
        Autoregressive gneratio with tempurature and top-k sampling 
        """
        for _ in range (max_new): 
            # context window croppping 
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:,-self.config.n_ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]/temperature 

            if top_k is not None: 
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits< v[:,[-1]]] = float('-inf')

            probs = F.softmax(logits,dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx,next_id],dim=1)
        return idx 












class MLP(nn.Module): 
    """
    Feed forward network
    Hidden dimension set in config

    """
    def __init__(self,config: GPTConfig):
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
    def __init__(self,ndim,bias:bool=True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None 
        self.eps = eps 

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
        self.use_flash = config.flash 
        self.is_sparse = ( layer_idx % 2 ==1) # odd layers are sparse 
        self.sparse_block = config.sparse_block_size 

        # fused QKV projection 
        self.qkv = nn.Linear(config.n_embd, 3*config.n_embd, bias = config.bias) 
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias) 
        self.attn_drop = nn.Dropout(config.dropout) 
        self.resid_drop = nn.Dropout(config.dropout) 

        # causal mask (used only when Flash attention is disabled )
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                       .view(1,1,config.n_ctx,config.n_ctx)
        )

    def _sparse_mask(self, T:int, device: torch.device) -> torch.Tensor: 
        """
        boolean causal + local-band mask for sparse layers 
        """
        causal = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool))
        band = torch.tril(torch.ones(T,T,device=device, dtype=torch.bool), diagonal=0) & ~torch.tril(torch.ones(T,T,device = device, dtype=torch.bool),diagonal = -(self.sparse_block +1)) 
        mask = causal & band 
        return mask.view(1,1,T,T)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B,T,C = x.shape 

        #fused qkv split 
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        def reshape(t): 
            return t.view(B,T, self.head, self.head_dim).transpose(1,2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        if self.use_flash and not self.is_sparse: 
            # flahs attention path 
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask = None, 
                dropout_p = self.dropout if self.training else 0.0, 
                is_causal = True, 
            )
        else: 
            #manual path : used for sparse layers (and dense fallback)
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q@k.transpose(-2,-1))*scale

            if self.is_sparse: 
                mask = self._sparse_mask(T,x.device)
                att = att.masked_fill(~mask, float('-inf'))
            else: 
                att = att.masked_fill(
                    self.causal_mask[:,:,:T,:T] == 0, float('-inf')
                )

            att = F.softmax(att,dim=-1)
            att = self.attn_drop(att)
            y = att@v 
        #merge heads and project 
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.resid_drop(self.proj(y))
    
    









class Block(nn.Module): 

    def __init__(self,config:GPTConfig,layer_idx:int): 
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd,bias=config.bias)
        self.attn = CausalSelfAttention(config,layer_idx) 
        self.ln_2 = LayerNorm(config.n_embd,bias=config.bias)
        self.mlp = MLP(config) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = x+ self.attn(self.ln_1(x)) # attention residual 
        x = x+ self.mlp(self.ln_2(x)) # MLP residual
