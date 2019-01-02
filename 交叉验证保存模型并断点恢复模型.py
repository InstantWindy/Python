
# coding: utf-8

# In[1]:

import os
os.environ['TF_CPP_MIN_LOGLEVEL']="2"
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"


# In[10]:


import logging
from torch import nn
from torch import optim
# from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from pspnet_Copy1 import PSPNet
from dataset import ADE20KLoader
from torchvision.transforms import Compose,ToTensor,Normalize
from augmentation import Scale,RandomRotation,CenterCrop,RandomHorizontalFlip,ToLabel
from metrics import runningScore


# In[11]:

num_classes=151
batch_size=4
models = {
    'squeezenet': lambda: PSPNet(num_classes,sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(num_classes,sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(num_classes,sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(num_classes,sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(num_classes,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(num_classes,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(num_classes,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


# In[12]:

#snapshot存放的是预训练的权重
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
#         _, epoch = os.path.basename(snapshot).split('_')
        epoch = 12
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


# In[13]:

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,                                               
                            max_iter=100, power=0.9):
    if iter % lr_decay_iter or iter > max_iter:#每lr_decay_iter下降，不等于0时返回
        return optimizer

    lr = init_lr*(1 - float(iter)/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


# # 加载数据

# In[14]:

input_transform=Compose([
    Scale((256,256),Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
    
])
target_transform=Compose([
     Scale((256,256),Image.NEAREST),
     ToLabel()
])
data_augs=Compose([
    RandomHorizontalFlip(),
    RandomRotation(),
])
train_loader=DataLoader(ADE20KLoader("/home/lulu/FCN_VGG19/ADEChallengeData2016/",split='training',
                                    input_transform=input_transform,
                                    target_transform=target_transform,augamentation=data_augs),num_workers=2,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(ADE20KLoader("/home/lulu/FCN_VGG19/ADEChallengeData2016/",split='validation',
                                    input_transform=input_transform,
                                    target_transform=target_transform,augamentation=data_augs),num_workers=2,batch_size=batch_size,shuffle=True)





def train( models_path, backend, snapshot, alpha, epochs, init_lr, ):
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network(snapshot, backend)
#     net.train()
   
    models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
 
    class_weights = torch.ones(num_classes).cuda()
    class_weight=torch.ones(batch_size,num_classes).cuda()
    
    optimizer = optim.Adam(net.parameters(), lr=start_lr,weight_decay=0.0001)
     # Setup Metrics
    running_metrics = runningScore(num_classes)

    best_iou = -100.0
	#从断点出恢复训练
    for epoch in range(starting_epoch, starting_epoch + epochs):
        
        seg_criterion = nn.NLLLoss2d(weight=class_weights)
#         cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)#二分类
        
        epoch_losses = []
        train_iterator = tqdm(train_loader, total=len(train_loader))
        
        net.train()
        for x, y, y_cls in train_iterator:

            optimizer.zero_grad()
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).cuda()
#             #y:torch.Size([16, 1, 256, 256])


            out, out_cls = net(x)
#             print('out_cls:',out_cls.size())#16,150,256,256

            

            seg_loss = seg_criterion(out, y.squeeze(1))
            cls_loss = seg_criterion(out_cls, y.squeeze(1))
            
            loss = seg_loss + alpha * cls_loss

            epoch_losses.append(loss.data[0])

            status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, '.format(epoch + 1, loss.data[0], np.mean(epoch_losses))
            train_iterator.set_description(status)#tadm中可以打印信息
           
            loss.backward()
            optimizer.step()
            
        net.eval()
        for i_val, (images_val, labels_val,label_cls) in tqdm(enumerate(val_loader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            outputs,outputs_cls = net(images_val)#outputs=batch,num_classes,H,W
           
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        running_metrics.reset()
        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            print("{}_{}_best_model.pkl".format(os.path.join(models_path,'PSPNet'), 'ADEK'))
            torch.save(net.state_dict(), "{}_{}_best_model.pkl".format(os.path.join(models_path,'PSPNet'), 'ADEK'))

        
        
        poly_lr_scheduler(optimizer, init_lr,epoch , lr_decay_iter=10,max_iter=100, power=0.9)
#         torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
 



#定义参数
models_path='checkpoint'
backend='resnet101'
snapshot='checkpoint/PSPNet_ADEK_best_model.pkl'

alpha=0.4
epochs=88
start_lr=0.01

if __name__ == '__main__':

    train(models_path,backend,snapshot,alpha,epochs,start_lr)






