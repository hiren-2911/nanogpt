import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
torch.manual_seed(1337)


batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 200
n_embed=384
n_head=6
n_layer=6
dropout=0.2
device = 'mps'

with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()
#print(f"length of text is:{len(text)}")


chars=sorted(list(set(text)))
vocab_size=len(chars)
#print(f'Vocab is:{"".join(chars)}') #first characted is next line character
#print(f'Length of vocab is:{vocab_size}')


stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode = lambda l:"".join(itos[s] for s in l)

data=torch.tensor(encode(text), dtype=torch.long) 


n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]


def get_batch(split):
    data=train_data if split=="train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))  #generates random indices for given batch size batch size
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y


# class LayerNorm(nn.Module):
#     def __init__(self, dim, eps=1e-5):
#         super().__init__()
#         self.eps=eps
#         self.gamma=torch.ones(dim)
#         self.beta = torch.ones(dim)

#         def __call__(self,x):
#             xmean=x.mean(1, keepdim=True)
#             xvar=x.var(1,keepdim=True)
#             xhat=(x-xmean)/torch.sqrt(xvar+self.eps)
#             self.out=self.gamma*xhat + self.beta
#             return self.out



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.val = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) #buffers are used when we want to use non learnable variables in Pytorch.
        # if we use normal variables, they will be learnable, so we use buffer variables.
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C= x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.val(x)

        wei= q @ k.transpose(-2, -1) * C**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        out = wei @ v 
        return out
    

class MultipleHEadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head) :
        super().__init__()
        head_size=n_embed//n_head
        self.sa=MultipleHEadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x= x + self.ffwd(self.ln2(x))
        return x





class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embed)  #learning parameter or layer
        self.position_embedding_table=nn.Embedding(block_size,n_embed)
        # self.sa_heads = MultipleHEadAttention(4, n_embed//4)
        # self.ffwd = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head=nn.Linear(n_embed,vocab_size)

    def forward(self, idx, targets=None):
        B,T=idx.shape
        token_emb=self.token_embedding_table(idx) #Batch, Time, n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        logits = self.lm_head(x)  #Batch, Time, Vocab_size
        if targets is None:
            loss=None
        else:
            B,T,C = logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next), dim=1)
        return idx

model=BigramLanguageModel().to(device)

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()

    for split in ['train','val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)

for iter in range(max_iters):
    if iter%eval_interval==0:
        losses=estimate_loss()
        print(f'step {iter}: train loss:{losses["train"]:.4f},val loss:{losses["val"]:.4f}')

    xb,yb=get_batch('train')
    # xb=xb.to(device)
    # yb=yb.to(device)
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter%5==0:
        print(f'{iter} Done')



context=torch.zeros((1,1),dtype=torch.long, device=device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))