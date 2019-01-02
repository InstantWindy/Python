import numpy as np
import torch
import torch.nn as nn
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
from torch.nn import Parameter
# 定义 bilinear kernel
lr_pow=0.9
learning_rate=1e-3


class PolyDecay:
    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs
    
    def scheduler(self, epoch,optimizer):
        lr=self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)
        optimizer.param_groups[0]['lr']=lr
#         return self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)
            
class ExpDecay:
    def __init__(self, initial_lr, decay):
        self.initial_lr = initial_lr
        self.decay = decay
    
    def scheduler(self, epoch,optimizer):
        lr=self.initial_lr * np.exp(-self.decay*epoch)
        optimizer.param_groups[0]['lr']=lr
        
#         return self.initial_lr * np.exp(-self.decay*epoch)



class GroupBatchnorm2d(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupBatchnorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
    
    
class GroupNormMoving(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5,
                 momentum=0.1, affine=True,
                 track_running_stats=True
                 ):
        super(GroupNormMoving, self).__init__()

        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps

        self.momentum = momentum
        self.affine = affine

        self.track_running_stats = track_running_stats

        tensor_shape = (1, num_features, 1, 1)

        if self.affine:
            self.weight = Parameter(torch.Tensor(*tensor_shape))
            self.bias = Parameter(torch.Tensor(*tensor_shape))
           
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            #     self.register_buffer('running_mean', torch.zeros(*tensor_shape))
            #     self.register_buffer('running_var', torch.ones(*tensor_shape))
            # else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0, "Channel must be divided by groups"

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        if self.running_mean is None or self.running_mean.size() != mean.size():
            # self.running_mean = Parameter(torch.Tensor(mean.data.clone()))
            # self.running_var = Parameter(torch.Tensor(var.data.clone()))
        
            self.running_mean = Parameter(torch.cuda.FloatTensor(mean.data))
            self.running_var = Parameter(torch.cuda.FloatTensor(var.data))
           

        if self.training and self.track_running_stats:
            self.running_mean.data = mean.data * self.momentum + \
                                     self.running_mean.data * (1 - self.momentum)
            self.running_var.data = var.data * self.momentum + \
                                    self.running_var.data * (1 - self.momentum)

        # mean = self.running_mean
        # var = self.running_var

        x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        x = x.view(N, C, H, W)
       
        return x * self.weight + self.bias

    def reset_parameters(self):
        if self.track_running_stats:
            if self.running_mean is not None and self.running_var is not None:
                self.running_mean.zero_()
                self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))
    
    
    
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter,max_iter):
    lr = lr_poly(learning_rate, i_iter, max_iter, lr_pow)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
        
     
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()#使用BytesIO操作二进制数据
    return PIL.Image.fromarray(a) #.save(f, fmt)
    #display(Image(data=f.getvalue()))#获取写入的数据


def showtensor(a):
    #参数a是numpy类型
    mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    inp = a[ :, :, :]
   # inp = a[0, :, :, :]
    inp = inp.transpose(1, 2, 0)
    inp = std * inp + mean
    inp *= 255
    return showarray(inp)
    #clear_output(wait=True)#Clear the output of the current cell receiving output.
