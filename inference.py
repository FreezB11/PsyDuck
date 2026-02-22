import torch 
# from model import *
from model import LModel   # IMPORTANT
import tiktoken
from config import *

encoding  = tiktoken.get_encoding("cl100k_base")

model = LModel().to(device)
model = torch.load("model-ckpt.pt")
# model.load_state_dict(torch.load("214M-4.6.pt", map_location=device))
model.eval()

start = 'The'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_token=100, temperature=0.7, top_k=10)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')