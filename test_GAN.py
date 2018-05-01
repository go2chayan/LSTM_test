import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

'''
Sample code of a Generative Adversarial Network
'''

def get_norm_dist(mu,sigma):
    '''
    This is the real data generator. We assume the real data is
    just a normal distribution with specified mean and variance.
    '''
    return lambda n:torch.Tensor(np.random.normal(mu,sigma,(1,n)))

def get_generator_input_sampler():
    '''
    This is the random input maker for the generator.Let's assume
    it is just a uniform random generator.
    '''
    return lambda n:torch.rand(1,n)

class Generator(nn.Module):
    '''
    This is the generator model.
    '''
    def __init__(self,input_size,hidden_size,output_size):
        super(Generator,self).__init__()
        # hidden layer 1. Elu nonlinearity
        self.map1 = nn.Linear(input_size,hidden_size)
        # hidden layer 2. Sigmoid nonlinearity
        self.map2 = nn.Linear(hidden_size,hidden_size)
        # Output layer.
        self.map3 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    '''
    This is the discriminator module. It's going to get samples
    from either the real data generator or the generator and will
    output 0 or 1 to represent 'fake' or 'real' respectively.
    '''
    def __init__(self,input_size,hidden_size,output_size):
        super(Discriminator,self).__init__()
        # hidden layer 1.
        self.map1 = nn.Linear(input_size,hidden_size)
        # hidden layer 2.
        self.map2 = nn.Linear(hidden_size,hidden_size)
        # Output layer.
        self.map3 = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

def train():
    inp_size = 10
    # Real dataset, a normal distribution
    R = get_norm_dist(-1,0.5)
    # Generator input sampler. Uniform
    I = get_generator_input_sampler()
    # Generator
    G = Generator(inp_size,10,inp_size)
    g_optim = optim.Adam(G.parameters(),lr=0.01)
    # Discriminator
    D = Discriminator(inp_size,10,1)
    d_optim = optim.Adam(D.parameters(),lr=0.01)
    # Discriminator Loss Function
    lossfn = nn.BCELoss()

    # Start training
    for i in range(800):
        # Train the discriminator with some true data
        for d_idx in range(30):
            D.zero_grad()
            # Real data
            real_data = Variable(R(inp_size))
            d_output = D(real_data)
            loss1 = lossfn(d_output,Variable(torch.ones(1)))
            #d_optim.step()
            # Artificially generated data
            gen_input = Variable(I(inp_size))
            generated_data = G(gen_input).detach()
            d_output = D(generated_data)
            d_loss = lossfn(d_output,Variable(torch.zeros(1)))+loss1
            d_loss.backward()
            d_optim.step()
        # Train the generator to fool the discriminator
        for g_idx in range(10):
            G.zero_grad()
            # Artificially generated data
            gen_input = Variable(I(inp_size))
            generated_data = G(gen_input)
            d_output = D(generated_data)
            g_loss = lossfn(d_output,Variable(torch.ones(1)))
            g_loss.backward()
            g_optim.step()
        print 'iteration:',i,'D_loss:',d_loss.data[0],'G_loss:',g_loss.data[0],\
            'mean of Gen. data:',generated_data.data.numpy().mean(),\
            'std of Gen. data:',generated_data.data.numpy().std()

