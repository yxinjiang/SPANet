import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(1)

from collections import OrderedDict
# import common

from torch.autograd import Variable

######print network
def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

###### Layer 
def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,):
        super(Bottleneck,self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False,dilation=2)
        m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu= nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x) 
        return out


class Renet(nn.Module):
    def __init__(self, in_channels):
        super(Renet, self).__init__()
        self.in_channel = in_channels
        self.out_channel = in_channels
        self.hidden_size = in_channels
        self.vertical = nn.LSTM(input_size=in_channels, hidden_size=in_channels, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=in_channels, hidden_size=in_channels, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv_down = nn.Conv2d(in_channels, self.out_channel, 1)
        self.conv_up = nn.Conv2d(in_channels, self.out_channel, 1)
        self.conv_right = nn.Conv2d(in_channels, self.out_channel, 1)
        self.conv_left = nn.Conv2d(in_channels, self.out_channel, 1)

    def forward(self, input):
        x = input      
        

        if len(x.shape) == 3:
            _,height,width = x.shape
        elif len(x.shape) ==4:
            _,_,height,width = x.shape
        else:
            print("Error")
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        #print('Renet input shape: ',x.shape)
            #temp = []
        temp_up=[]
        temp_down=[]
        for i in range(height):
            h, _ = self.vertical(x[:, :, i, :])# h.shape: batch, width, 512
            #temp.append(h)  
            temp_up.append(h[:,:,:self.hidden_size])
            temp_down.append(h[:,:,self.hidden_size:])
        #x = torch.stack(temp, dim=2)  # batch, width, height, 512
        #x = torch.transpose(x, 1, 3)  # batch, 512, height, width
        x_down = torch.stack(temp_down,dim=2)
        x_down = torch.transpose(x_down,1,3)
        x_down = self.conv_down(x_down)

        x_up = torch.stack(temp_up,dim=2)
        x_up = torch.transpose(x_up,1,3)
        x_up = self.conv_up(x_up)

        
        #temp = []
        temp_right=[]
        temp_left=[]
        for i in range(width):
            h, _ = self.horizontal(x[:, i, :, :])
            #temp.append(h)  # batch, width, 512
            temp_right.append(h[:,:,:self.hidden_size])
            temp_left.append(h[:,:,self.hidden_size:])
        #x = torch.stack(temp, dim=3)  # batch, height, 512, width
        # x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        #x = self.conv(x)
        x_right = torch.stack(temp_right,dim=2)
        x_right = torch.transpose(x_right,1,3)
        x_right = self.conv_right(x_right)
        #print('x_right shape: ',x_right.shape)

        x_left = torch.stack(temp_left,dim=2)
        x_left = torch.transpose(x_left,1,3)
        x_left = self.conv_left(x_left)
        #print('x_left shape: ',x_left.shape)

        
        return x_down,x_up,x_right,x_left

class Spacial_LSTM(nn.Module):
    def __init__(self,in_channels):
        super(Spacial_LSTM,self).__init__()
        self.feature_map  = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        #self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        #self.up_weight    = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        #self.down_weight  = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        #weight = nn.Parameter(self.left_weight.weight)
        #self.right_weight.weight = weight
        #self.up_weight.weight = weight
        #self.down_weight = weight

        #bias = nn.Parameter(self.left_weight.bias)
        #self.right_weight.bias = bias
        self.renet = Renet(in_channels)
               
    def forward(self,input):
        #down = self.left_weight(input)
        #up = self.right_weight(input)
        #right = self.right_weight(input)
        #left = self.left_weight(input)
        #print('Spacial_LSTM input shape: ',input.shape)
        x = self.feature_map(input)
        #print('feature_map output shape: ',x.shape)
        up_weight,down_weight,right_weight,left_weight = self.renet(x)
        return up_weight,right_weight,down_weight,left_weight   


class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1):
        super(SAM,self).__init__()
        self.out_channels = out_channels
        #self.irnn1 = Spacial_IRNN(self.out_channels)
        #self.irnn2 = Spacial_IRNN(self.out_channels)
        self.irnn1 = Spacial_LSTM(self.out_channels)
        self.irnn2 = Spacial_LSTM(self.out_channels)
        self.conv_in = conv3x3(in_channels,in_channels)
        self.conv2 = conv3x3(in_channels*4,in_channels)
        self.conv3 = conv3x3(in_channels*4,in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels,1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        #print("conv_in out shape: ",out.shape)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            #print('top_up shape ',top_up.shape)
            #print('weight[:,0:1,:,:] shape ',weight[:,0:1,:,:].shape)
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            #print('top_left shape ',top_left.shape)
            #print('weight[:,3:4,:,:] shape ',weight[:,3:4,:,:].shape)
            top_left.mul(weight[:,3:4,:,:])
            
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.sigmod(self.conv_out(out))
        return mask

###### Network
class SPANet(nn.Module):
    def __init__(self):
        super(SPANet,self).__init__()

        self.conv_in = nn.Sequential(
            conv3x3(3,32),
            nn.ReLU(True)
            )
        self.SAM1 = SAM(32,32,1)
        self.res_block1 = Bottleneck(32,32)
        #self.res_block2 = Bottleneck(32,32)
        #self.res_block3 = Bottleneck(32,32)
        self.res_block4 = Bottleneck(32,32)
        self.res_block5 = Bottleneck(32,32)
        self.res_block6 = Bottleneck(32,32)
        #self.res_block7 = Bottleneck(32,32)
        #self.res_block8 = Bottleneck(32,32)
        #self.res_block9 = Bottleneck(32,32)
        #self.res_block10 = Bottleneck(32,32)
        #self.res_block11 = Bottleneck(32,32)
        #self.res_block12 = Bottleneck(32,32)
        #self.res_block13 = Bottleneck(32,32)
        #self.res_block14 = Bottleneck(32,32)
        #self.res_block15 = Bottleneck(32,32)
        #self.res_block16 = Bottleneck(32,32)
        self.res_block17 = Bottleneck(32,32)
        self.conv_out = nn.Sequential(
            conv3x3(32,3)
        )
    def forward(self, x):

        out = self.conv_in(x)
        out = F.relu(self.res_block1(out) + out)
        #out = F.relu(self.res_block2(out) + out)
        #out = F.relu(self.res_block3(out) + out)
        
        Attention1 = self.SAM1(out) 
        out = F.relu(self.res_block4(out) * Attention1  + out)
        out = F.relu(self.res_block5(out) * Attention1  + out)
        out = F.relu(self.res_block6(out) * Attention1  + out)
        
        #Attention2 = self.SAM1(out) 
        #out = F.relu(self.res_block7(out) * Attention2 + out)
        #out = F.relu(self.res_block8(out) * Attention2 + out)
        #out = F.relu(self.res_block9(out) * Attention2 + out)
        
        #Attention3 = self.SAM1(out) 
        #out = F.relu(self.res_block10(out) * Attention3 + out)
        #out = F.relu(self.res_block11(out) * Attention3 + out)
        #out = F.relu(self.res_block12(out) * Attention3 + out)
        
        #Attention4 = self.SAM1(out) 
        #out = F.relu(self.res_block13(out) * Attention4 + out)
        #out = F.relu(self.res_block14(out) * Attention4 + out)
        #out = F.relu(self.res_block15(out) * Attention4 + out)
        
        #out = F.relu(self.res_block16(out) + out)
        out = F.relu(self.res_block17(out) + out)
       
        out = self.conv_out(out)

        return Attention1 , out

if __name__ == "__main__":
    random_input = Variable(torch.FloatTensor(1,3,32,32).normal_(),requires_grad=False) #batch,in_channel,height,width 
    spanet = SPANet()
    print_network(spanet)
    out = spanet(random_input)

    
