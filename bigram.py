import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
torch.manual_seed(1337)


batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200



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
    return x,y



class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)  #learning parameter or layer

    def forward(self, idx, targets=None):
        logits=self.token_embedding_table(idx) #Batch, Time, Channel
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
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next), dim=1)
        return idx

model=BigramLanguageModel(vocab_size)

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
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



context=torch.zeros((1,1),dtype=torch.long)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))