import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np

class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor,
                          bn_weights, bn_biases) -> torch.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.
    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x

def conv_block(in_channels: int, out_channels: int, kernel=3, pad=1) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=pad),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def conv_block_no_pool(in_channels: int, out_channels: int, kernel=3, pad=1) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=pad),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


##########
# Models #
##########
def get_few_shot_encoder(num_input_channels=1, kernel=3, pad=1, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64 + covariance_matrix_dim, kernel, pad),
        Flatten(),
    )

def get_few_shot_encoder_more_channels(num_input_channels=1, kernel=3, pad=1, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 32, kernel, pad),
        conv_block(32, 64, kernel, pad),
        conv_block(64, 96, kernel, pad),
        conv_block(96, 128 + covariance_matrix_dim, kernel, 0),
        Flatten(),
    )

def get_few_shot_encoder_more_layers(num_input_channels=1, kernel=3, pad=1, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block_no_pool(64, 64, kernel, pad),
        conv_block_no_pool(64, 64 + covariance_matrix_dim, kernel, pad),
        Flatten(),
    )

def get_few_shot_encoder_v2(num_input_channels=1, kernel=3, pad=1, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        Flatten(),
        nn.Linear(576,576 + covariance_matrix_dim),
        Flatten()
    )

def get_few_shot_encoder_linear(final_output=1152,num_input_channels=1, kernel=3, pad=1, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        Flatten(),
        nn.Linear(576,final_output)
    )

def get_few_shot_encoder_fcnn(final_output=1152,num_input_channels=1, kernel=3, pad=1, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 128, kernel, pad),
        nn.Conv2d(128, final_output, 1),
        nn.AvgPool2d(3),
        Flatten()
    )

def get_few_shot_encoder_fcnn_v2(final_output=1152,num_input_channels=1, kernel=3, pad=1, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 64, kernel, pad),
        conv_block(64, 128, kernel, pad),
        nn.Conv2d(128, 128, 1),
        Flatten()
    )

def get_few_shot_encoder_large(num_input_channels=1, kernel=3, pad=0, covariance_matrix_dim = 0) -> nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks
    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(
        conv_block(num_input_channels, 64, 7, pad),
        conv_block(64, 96, kernel, pad),
        conv_block(96, 96, kernel, pad),
        conv_block(96, 64 + covariance_matrix_dim, kernel, pad),
        Flatten(),
    )

def conv(kernel_size, stride, in_size, out_size,pad=1):
    layer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=pad),
            #nn.BatchNorm2d(out_size, momentum=0.01, eps=0.001),
            nn.BatchNorm2d(out_size),
            #nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
    return layer

class FewShotClassifier(nn.Module):
    def __init__(self, num_input_channels, k_way, final_layer_size = 64):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(FewShotClassifier, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        self.logits = nn.Linear(final_layer_size, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""

        for block in [1, 2, 3, 4]:
            x = functional_conv_block(x, weights[f'conv{block}.0.weight'], weights[f'conv{block}.0.bias'],
                                      weights.get(f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])

        return x

class ReptileModel(nn.Module):
    """
    CNN for Mini-Imagenet meta learning.
    """
    def __init__(self, num_classes, num_input_channels, output_size, num_filters):
        super(ReptileModel, self).__init__()
        kernel_size = 3
        dims = 84 # Mini 

        self.layer1 = conv(kernel_size, 1, num_input_channels, num_filters)
        self.layer2 = conv(kernel_size, 1, num_filters, num_filters) 
        self.layer3 = conv(kernel_size, 1, num_filters, num_filters)
        self.layer4 = conv(kernel_size, 1, num_filters, num_filters)

        self.final = nn.Linear(output_size, num_classes)
        #self.final = nn.Linear(num_filters * 5 * 5, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        return self.final(x)

class ReptileModel_96p(nn.Module):
    """
    CNN for Mini-Imagenet meta learning.
    """
    def __init__(self, num_classes, num_input_channels, output_size, num_filters):
        super(ReptileModel_96p, self).__init__()
        kernel_size = 3
        dims = 84 # Mini 

        self.layer1 = conv(7, 1, num_input_channels, num_filters, pad=0)
        self.layer2 = conv(kernel_size, 1, num_filters, num_filters, pad=0) 
        self.layer3 = conv(kernel_size, 1, num_filters, num_filters, pad=0)
        self.layer4 = conv(kernel_size, 1, num_filters, num_filters, pad=0)

        self.final = nn.Linear(output_size, num_classes)
        #self.final = nn.Linear(num_filters * 5 * 5, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        return self.final(x)
    

class ReptileModel_192p(nn.Module):
    """
    CNN for Mini-Imagenet meta learning.
    """
    def __init__(self, num_classes, num_input_channels, output_size, num_filters):
        super(ReptileModel_192p, self).__init__()
        kernel_size = 3
        dims = 84 # Mini 

        self.layer1 = conv(7, 1, num_input_channels, num_filters, pad=0)
        self.layer2 = conv(kernel_size, 1, num_filters, num_filters, pad=0) 
        self.layer3 = conv(kernel_size, 1, num_filters, num_filters, pad=0)
        self.layer4 = conv(kernel_size, 1, num_filters, num_filters, pad=0)
        self.layer5 = conv(kernel_size, 1, num_filters, num_filters, pad=0)
        
        self.final = nn.Linear(output_size, num_classes)
        #self.final = nn.Linear(num_filters * 5 * 5, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)
        return self.final(x)
    

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_input_channels=1):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(num_input_channels,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out
    
class OmniglotNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(OmniglotNet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('block1', conv_block(x_dim, hid_dim)),
            ('block2', conv_block(hid_dim, hid_dim)),
            ('block3', conv_block(hid_dim, hid_dim)),
            ('block4', conv_block(hid_dim, z_dim)),
        ]))

    def forward(self, x, weights=None):
        if weights is None:
            x = self.encoder(x)
        else:
            x = F.conv2d(x, weights['encoder.block1.conv.weight'], weights['encoder.block1.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block1.bn.weight'], bias=weights['encoder.block1.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block2.conv.weight'], weights['encoder.block2.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block2.bn.weight'], bias=weights['encoder.block2.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block3.conv.weight'], weights['encoder.block3.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block3.bn.weight'], bias=weights['encoder.block3.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block4.conv.weight'], weights['encoder.block4.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block4.bn.weight'], bias=weights['encoder.block4.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
        return x.view(x.size(0), -1)
    
class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-self.dilation] # TODO: make this correct for different strides/padding

class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg) # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)
        
class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** (i+1), filters)
                                           for i in range(int(math.ceil(math.log(seq_length))))])

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)
        return torch.transpose(input, 1, 2)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size, use_cuda=True):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        self.use_cuda = use_cuda
        
    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        if self.use_cuda:
            mask = torch.ByteTensor(mask).cuda()
        else:
            mask = torch.ByteTensor(mask)

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2) # shape: (N, T, in_channels + value_size)
    
class SnailFewShot(nn.Module):
    def __init__(self, N, K, task, use_cuda=True):
        # N-way, K-shot
        super(SnailFewShot, self).__init__()
        if task == 'omniglot':
            self.encoder = OmniglotNet()
            num_channels = 64 + N
        elif task == 'mini_imagenet':
            self.encoder = MiniImagenetNet()
            num_channels = 384 + N
        else:
            raise ValueError('Not recognized task value')
        num_filters = int(math.ceil(math.log(N * K + 1)))
        self.attention1 = AttentionBlock(num_channels, 64, 32, use_cuda)
        num_channels += 32
        self.tc1 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128, use_cuda)
        num_channels += 128
        self.tc2 = TCBlock(num_channels, N * K + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256, use_cuda)
        num_channels += 256
        self.fc = nn.Linear(num_channels, N)
        self.N = N
        self.K = K
        self.use_cuda = use_cuda

    def forward(self, input, labels):
        x = self.encoder(input)
        batch_size = int(labels.size()[0] / (self.N * self.K + 1))
        last_idxs = [(i + 1) * (self.N * self.K + 1) - 1 for i in range(batch_size)]
        if self.use_cuda:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).cuda()
        else:
            labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1])))
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, self.N * self.K + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        return x
    
    
def conv_block_residual(in_channels: int, out_channels: int, relu=True) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    if relu:
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    else:
        return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels)
    )
    
def residual_block(in_channels: int, out_channels: int) -> nn.Module:
    
    return nn.Sequential(
        conv_block_residual(in_channels, out_channels, True),
        conv_block_residual(out_channels, out_channels, True),
        conv_block_residual(out_channels, out_channels, False)
    )

def reluAndmaxPool():
    return nn.Sequential(nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

class EResnet_model(nn.Module):
    """
    CNN for Mini-Imagenet meta learning.
    """
    def __init__(self):
        super(EResnet_model, self).__init__()
        
        self.one = nn.Conv2d(3,64,1)
        self.block1 = residual_block(3, 64)
        self.relu1 = reluAndmaxPool()
        
        self.two = nn.Conv2d(64,96,1)
        self.block2 = residual_block(64, 96)
        self.relu2 = reluAndmaxPool()
        
        self.three = nn.Conv2d(96,128,1)
        self.block3 = residual_block(96, 128)
        self.relu3 = reluAndmaxPool()
        
        self.block4 = residual_block(128, 128)
        self.relu4 = reluAndmaxPool()
        
        self.endConv = nn.Conv2d(128, 1024, 1)
        self.GAP = nn.AvgPool2d(1024)
        
        self.flatten = Flatten()
        
    def forward(self, x):
        
        residual1 = self.one(x)
        x = self.block1(x)
        x += residual1
        x = self.relu1(x)

        residual2 = self.two(x)
        x = self.block2(x)
        x += residual2
        x = self.relu2(x)
        
        residual3 = self.three(x)
        x = self.block3(x)
        x += residual3
        x = self.relu3(x)
        
        residual4 = x
        x = self.block4(x)
        sh = residual4.shape
        x += residual4
        x = self.relu4(x)
        
        x = self.endConv(x)
        
        x = self.GAP(x)
        x = self.flatten(x)
        
        return x
    
class SoftplusTrainable(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.offset = torch.nn.Parameter(torch.ones(1, 1))
        self.scale = torch.nn.Parameter(torch.ones(1, 1))
        self.div = torch.nn.Parameter(torch.ones(1, 1))
        self.eps = 1e-10
        
    def forward(self, X):
        
        inv_covariance_matrix = self.offset.pow(2) + self.scale.pow(2)*F.softplus((X/(self.div.pow(2) + self.eps)),beta=1, threshold=20)
        
        return inv_covariance_matrix

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'

        trunk = []

        indim = 3
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)
 
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

def ResNet10(flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)