#############################################显示图片##########################################

def show_images(images): # 定义画图工具,images:batch,channel,h,w
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0]))) #128
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1]))) #784

    ## 画图之前首先设置figure对象，此函数相当于设置一块自定义大小的画布，使得后面的图形输出在这块规定了大小的画布上，其中参数figsize设置画布大小
    fig = plt.figure(figsize=(sqrtn, sqrtn)) 
    #指定子图将放置的网格的几何位置。 需要设置网格的行数和列数
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
    
        #subplot()是将整个figure均等分割
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 

#############################################处理图片##########################################

def preprocess_img(x):
    x = tfs.ToTensor()(x)
    return (x - 0.5) / 0.5

def deprocess_img(x):
    return (x + 1.0) / 2.0

##############################################读文件############################################

with open('') as f:
	for c in f:
		print(c)#可以输出文本中的内容，一行一行的输出

##############################################poly学习率下降############################################

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,                                               
                            max_iter=100, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
	if iter % lr_decay_iter or iter > max_iter:
		return optimizer

	lr = init_lr*(1 - iter/max_iter)**power
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return lr


##############################################模型############################################

inception = models.inception_v3(pretrained=True)
self.feature = nn.Sequential(*list(inception.children())[:-1])
self.feature._modules.pop('13')#弹出第13个模块
self.feature.add_module('global average', nn.AvgPool2d(35))

torch.FloatTensor()是创建一个空张量，什么内容都没有


##############################################加载图片############################################
import PIL.Image as Image
import torchvision.transforms as transforms

img_size = 512


def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()



如果一个变量调用了detach()函数，那么形成该变量结点的图就脱离了，梯度也就不会传递到该变量前面去了
a=np.ones((21,3))
idx=np.ones((32,32)).astype('uint8')
a[idx]#shape:32,32,3

import matplotlib.pyplot as plt
%matplotlib inline
#不用加show()就可以显示

import sys
sys.path.append('..')#加载上一层路径