import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


lstm = nn.LSTM(3,3)  #input dim is 3, output dim is 3
inputs = [torch.randn(1,3) for _ in range(5)]

#initialize the hidden state
hidden = (torch.randn(1,1,3),torch.randn(1,1,3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    print('input shape: ',i.view(1,1,-1).shape)
    out,hidden = lstm(i.view(1,1,-1),hidden)
    print("out shape: ",out.shape)
    print("hidden ", hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs,dim=1).view(1,len(inputs),-1)
print("inputs shape: ",inputs.shape)
hidden = (torch.randn(1,5,3),torch.randn(1,5,3))
out,hidden = lstm(inputs,hidden)
print("out shape: ",out.shape)
print(torch.squeeze(out,0))
print(hidden)
print('hidden[0] shape: ',hidden[0].shape)


input_dim = 5
hidden_dim = 10
n_layers = 1
lstm_layer = nn.LSTM(input_dim,hidden_dim,n_layers,batch_first=True,bidirectional=True)
batch_size = 1
seq_len = 1
inp = torch.randn(batch_size,seq_len,input_dim)
hidden_state = torch.randn(n_layers*2,batch_size,hidden_dim)
cell_state = torch.randn(n_layers*2,batch_size,hidden_dim)
hidden = (hidden_state,cell_state)

out,hidden = lstm_layer(inp,hidden)
print("output shape: ",out.shape)
print("output ",out)
print("hidden[0]: ", hidden[0])
print("hidden[0] shape : ", hidden[0].shape)

seq_len = 3
inp = torch.randn(batch_size,seq_len,input_dim)
hidden_state = torch.randn(n_layers*2,batch_size,hidden_dim)
cell_state = torch.randn(n_layers*2,batch_size,hidden_dim)
hidden = (hidden_state,cell_state)

out,hidden = lstm_layer(inp,hidden)
print("output shape: ",out.shape)
print("out 1:",out[:,-1,:hidden_dim])
print("out 2:",out[:,0,hidden_dim:])
print("hidden[0]: ", hidden[0])
print("hidden[0] shape : ", hidden[0].shape)
print("hidden[1]: ", hidden[1])

out = out.squeeze()[-1,:]
print(out.shape)
print(out)


import numpy as np 
import torch, torch.nn as nn
from torch.autograd import Variable
random_input = Variable(torch.FloatTensor(5, 1, 1).normal_(), requires_grad=False)
random_input[:, 0, 0]

#Initialize a Bidirectional GRU Layer
bi_grus = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=True)
#Initialize a GRU Layer ( for Feeding the Sequence Reversely)
reverse_gru = torch.nn.GRU(input_size=1, hidden_size=1, num_layers=1, batch_first=False, bidirectional=False)
reverse_gru.weight_ih_l0 = bi_grus.weight_ih_l0_reverse
reverse_gru.weight_hh_l0 = bi_grus.weight_hh_l0_reverse
reverse_gru.bias_ih_l0 = bi_grus.bias_ih_l0_reverse
reverse_gru.bias_hh_l0 = bi_grus.bias_hh_l0_reverse
#Feed Input Sequence into Both Networks
bi_output, bi_hidden = bi_grus(random_input)
reverse_output, reverse_hidden = reverse_gru(random_input[np.arange(4, -1, -1), :, :])
print(reverse_output[:, 0, 0])
print(bi_output[:, 0, 1])
print(reverse_hidden)
print(bi_hidden[1])

in_channel = 3
out_channel = 1
hidden_size = 256
random_input = Variable(torch.FloatTensor(1,3,32,32).normal_(),requires_grad=False) #batch,in_channel,height,width 
batch,in_channel,height,width = random_input.shape
print(random_input[0,0,:,:].shape)
print(random_input[0,0,:,:])

vertical = nn.LSTM(input_size=in_channel, hidden_size=hidden_size, batch_first=True,bidirectional=True)  # each row
conv = nn.Conv2d(512, out_channel*2, 1)
conv_down = nn.Conv2d(256, out_channel, 1)
conv_up = nn.Conv2d(256, out_channel, 1)
x = torch.transpose(random_input, 1, 3)# batch, width, height, in_channel
temp=[]
temp_down=[]
temp_up=[]
for i in range(height):
    h, _ = vertical(x[:, :, i, :])
    temp.append(h)  # batch, width, 512
    temp_down.append(h[:,:,:hidden_size])
    temp_up.append(h[:,:,hidden_size:])

x = torch.stack(temp, dim=2)  # batch, width, height, 512
x = torch.transpose(x, 1, 3)  # batch, 512, height, width
x_down = torch.stack(temp_down,dim=2)
x_down = torch.transpose(x_down,1,3)
x_up = torch.stack(temp_up,dim=2)
x_up = torch.transpose(x_up,1,3)

conv_down.weight=nn.Parameter(conv.weight[:,:hidden_size,:])
conv_down.bias = nn.Parameter(conv.bias)
conv_up.weight=nn.Parameter(conv.weight[:,hidden_size:,:])
conv_up.bias = nn.Parameter(conv.bias)
x = conv(x)
x_down = conv_down(x_down)
x_up = conv_up(x_up)

in_channels = 3
random_input = Variable(torch.FloatTensor(1,in_channels,32,32).normal_(),requires_grad=False) #batch,in_channel,height,width 

left_weight  = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=1,padding=0)
x = left_weight(random_input)

x = torch.FloatTensor([100, 200, 300]).view(1, -1, 1, 1)
x = Variable(x)

conv = nn.Conv2d(in_channels=3,
                 out_channels=6,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=3,
                 bias=False)

# Set Conv weight to [0, 1, 2, 3, 4 ,5]
conv.weight.data = torch.arange(6).view(-1, 1, 1, 1).type(torch.float)
output = conv(x)
print(output)