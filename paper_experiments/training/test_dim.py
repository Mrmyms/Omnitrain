import torch
import sys
sys.path.append('../../src')
from omnitrain.sparse_cfc import SparseCfC

d_in = 25
hidden_dim = 100
d_out = 2
adj = torch.ones(hidden_dim, d_in + hidden_dim)
model = SparseCfC(d_in, hidden_dim, d_out, adj)
model.eval()

x_t = torch.randn(1, 1, 25)
x_t_seq = x_t.expand(-1, 2, -1)
times = torch.randn(1, 2, 1)

a = model(x_t_seq, times)
print(a.shape)
