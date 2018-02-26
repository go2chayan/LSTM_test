import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.W_xi = nn.Linear(1,1)
        self.W_hi = nn.Linear(1,1)
        self.W_xf = nn.Linear(1,1)
        self.W_hf = nn.Linear(1,1)
        self.W_xg = nn.Linear(1,1)
        self.W_hg = nn.Linear(1,1)
        self.W_xo = nn.Linear(1,1)
        self.W_ho = nn.Linear(1,1)

    def forward(self,x,h,c):
        i = F.sigmoid(self.W_xi(x) + self.W_hi(h))
        f = F.sigmoid(self.W_xf(x) + self.W_hf(h))
        g = F.relu(self.W_xg(x) + self.W_hg(h))
        o = F.sigmoid(self.W_xo(x)+self.W_ho(h))
        c_ = f*c + i*g
        h_ = o * F.relu(c_)
        return h_,c_


model = LSTM()
optimizer = optim.Adam(model.parameters(),lr=0.01)
for i in range(5000):
    inp = np.random.rand(4)*100.
    
    h = Variable(torch.Tensor([[0.]]))
    c = Variable(torch.Tensor([[0.]]))
    output = Variable(torch.Tensor([[float(inp.sum())]]))
    model.zero_grad()
    for j in inp.tolist():
        inp = Variable(torch.Tensor([[j]]))
        h,c = model(inp,h,c)
        print 'inp:',inp
        print 'h:',h
    loss = (h-output)**2.
    print 'loss:',loss
    loss.backward()
    optimizer.step()

def test(inp,model):
    h = Variable(torch.Tensor([[0.]]))
    c = Variable(torch.Tensor([[0.]]))
    for i in inp:
        h,c = model(Variable(torch.Tensor([[i]])),h,c)
    print 'input:',inp
    print 'output:',h


test([1,2,3,4],model)
test([1,2,3,4,6,7,10],model)
test([10,20],model)
test([111,112,113,114],model)
test([7,8,9,10],model)
