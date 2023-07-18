from __future__ import print_function
from sklearn.linear_model import LogisticRegression
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import torchvision
import torchvision.transforms as transforms
import sys
import os, sys, tarfile, errno
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import CenterCrop, Resize,ToTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from PIL import Image

import numpy as np
class ConvRBM():

    def __init__(self, k,Hidden=64, learning_rate=1e-4, momentum_coefficient=0.9, weight_decay=0,num_visible=3*32*32, num_hidden=3*24*24,batch_size=64,  use_cuda=True):
        #self.num_visible = num_visible
        #self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.k = k #Step for CD
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda


        self.conv1_weights = nn.Parameter(torch.randn(Hidden,3,9,9))*(10**(-3))
        self.conv1_visible_bias = nn.Parameter(torch.randn(3))
        self.conv1_hidden_bias = nn.Parameter(torch.randn(Hidden))
        self.maxpool1 = nn.MaxPool2d(4,stride=4,return_indices=True)


        self.conv1_weights_momentum = torch.zeros(Hidden,3,9,9)
        self.conv1_visible_bias_momentum = torch.zeros(3)
        self.conv1_hidden_bias_momentum = torch.zeros(Hidden)
        self.upsample1 = nn.MaxUnpool2d(4, stride=4)


        if self.use_cuda:
            self.conv1_weights = self.conv1_weights.cuda()
            self.conv1_visible_bias = self.conv1_visible_bias.cuda()
            self.conv1_hidden_bias = self.conv1_hidden_bias.cuda()


            self.conv1_weights_momentum = self.conv1_weights_momentum.cuda()
            self.conv1_visible_bias_momentum = self.conv1_visible_bias_momentum.cuda()
            self.conv1_hidden_bias_momentum = self.conv1_hidden_bias_momentum.cuda()


    #隐藏层sample
    def sample_hidden(self, visible_probabilities):
        out1 = F.conv2d(visible_probabilities,weight=self.conv1_weights,bias=self.conv1_hidden_bias,padding=0)#out1.shape=(64,32,32,32)因为添加了一圈
        out2 = ConvRBM.relu1(out1)

        self.out1,self.out_idc = self.maxpool1(out2)#self.out1=(64,32,8,8)
        #print('out1', self.out1.shape)

        hidden_probabilities = self.out1 #hidden_probabilities=(64,32,8,8)


        return hidden_probabilities
    #显层sample
    def sample_visible(self, hidden_probabilities):

        de_output1 = self.upsample1(hidden_probabilities,self.out_idc)#de_output1=(64,32,32,32)

        de_output2 = F.conv_transpose2d(de_output1,weight=self.conv1_weights,bias=self.conv1_visible_bias,stride=1,padding=0) #deoutput1=(64,3,40,40),对应input

        self.de_output1 = ConvRBM.relu6(de_output2)
        #print(de_output2)
        #print(self.de_output1)
        visible_probabilities = self.de_output1 #visible_probabilities=(64,3,40,40)

        return visible_probabilities

    def contrastive_divergence(self, input_data):
        positive_hidden_probabilities = self.sample_hidden(input_data)  #positive_hidden_probabilities=(64,32,7,7)
        pose_scale = positive_hidden_probabilities.size()#(64,32,7,7)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(pose_scale)).float() #positive_hidden_activations=(64,32,8,8)


        positive_associations = torch.matmul(input_data.view(self.batch_size,3*36*36).t(), positive_hidden_activations.view(self.batch_size,pose_scale[1]*pose_scale[2]*pose_scale[3])) #Back_down?
        #positive_associations=(3072,3969)

        # Negative phase
        hidden_activations = positive_hidden_activations #(64,32,7,7)  反向输出

        for step in range(self.k):

            visible_probabilities = self.sample_visible(hidden_activations)  #(64,3,40,40)  显层，反向输出


            hidden_probabilities= self.sample_hidden(visible_probabilities)  #然后用反向的输出当成输入！ (64,32,8,8)
            neg_scale = hidden_probabilities.size() #(64,32,8,8)

            hidden_activations = (hidden_probabilities >= self._random_probabilities(neg_scale)).float()

        negative_visible_probabilities = visible_probabilities #64，3，40，40
        neg_vis_scale = negative_visible_probabilities.size()
        negative_hidden_probabilities = hidden_probabilities
        neg_hid_scale = negative_hidden_probabilities.size()
        #print(negative_hidden_probabilities.shape)
        negative_associations1 = torch.matmul(negative_visible_probabilities.view(self.batch_size,neg_vis_scale[1]*neg_vis_scale[2]*neg_vis_scale[3]).t(), negative_hidden_probabilities.view(self.batch_size,neg_hid_scale[1]*neg_hid_scale[2]*neg_hid_scale[3]))


        # Update
        #print(negative_associations1.shape,positive_associations.shape)
        self.conv1_weights_momentum *= self.momentum_coefficient

        self.conv1_weights_momentum += (1. / (16. * 49.)) * torch.sum((positive_associations.view(16, 49, 64, 3, 9, 9) - negative_associations1.view(16, 49, 64, 3, 9, 9)),dim=(0, 1))

        self.conv1_visible_bias_momentum *= self.momentum_coefficient

        vis_bm_tmp = torch.sum(input_data - negative_visible_probabilities, dim=(0,1,2,3))
        self.conv1_visible_bias_momentum += (1.0/(3.*36.*36.))*vis_bm_tmp

        self.conv1_hidden_bias_momentum *= self.momentum_coefficient
        #print(positive_associations.shape,negative_hidden_probabilities.shape)
        hidden_bm_tmp = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=(0, 2, 3))
        self.conv1_hidden_bias_momentum += (1.0/(64.*7.*7.))*hidden_bm_tmp

        batch_size = input_data.size(0)

        #self.conv1_weights += self.conv1_weights_momentum * self.learning_rate / batch_size
        self.conv1_weights = torch.add(self.conv1_weights,self.conv1_weights_momentum * self.learning_rate / batch_size)

        #self.conv1_visible_bias = self.conv1_visible_bias_momentum * self.learning_rate / batch_size
        #self.conv1_hidden_bias += self.conv1_hidden_bias_momentum * self.learning_rate / batch_size
        #self.conv1_weights -= self.conv1_weights * self.weight_decay  # L2 weight decay
        self.conv1_visible_bias = torch.add(self.conv1_visible_bias,self.conv1_visible_bias_momentum * self.learning_rate / batch_size)
        self.conv1_hidden_bias = torch.add(self.conv1_hidden_bias,self.conv1_hidden_bias_momentum * self.learning_rate / batch_size)
        #self.conv1_weights = torch.add(self.conv1_weights, -self.conv1_weights * self.weight_decay)

        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities

    def sampling_bernoulli(probs):
        # pdb.set_trace()
        return probs - torch.rand(probs.size()).cuda()

    def sampling_gaussian(probs):
        return probs + torch.rand(probs.size()).cuda()

    def relu6(x):
        return torch.clamp(x, min=0, max=6)

    def relu1(x):
        return torch.clamp(x, min=0, max=1)
def tensor_to_image(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype('uint8')
    image = Image.fromarray(image)
    return image
########## CONFIGURATION ##########
BATCH_SIZE = 64
INPUT_DIM = 32  # 28 x 28 images
HIDDEN_UNITS = 64*7*7 #32*14*14
CD_K = 1

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)
########## LOADING DATASET ##########
print('Loading dataset...')

# 定义数据集和数据加载器
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # 定义数据预处理
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # 定义训练集
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 定义训练集加载器
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # 定义测试集
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # 定义测试集加载器



########## TRAINING Convolutional RBM ##########
print('Training Convolutional RBM...')

convrbm = ConvRBM(k=CD_K, use_cuda=CUDA)
convrbm_weights= torch.load('pre_train_for_convrbm64.pth')
#print(convrbm_weights)

print('++++++++++++++')
#convrbm_bias=torch.load('conv1_hidden_bias_in.pth')
convrbm.conv1_weights=convrbm_weights
print(torch.mean(convrbm.conv1_weights))
#print(torch.mean(convrbm.conv1_weights)) #0.0441,27%
#convrbm.conv1_hidden_bias=convrbm_bias
EPOCHS=0
for epoch in range(EPOCHS):
    epoch_error = 0.0
    count = 0
    for i,(batch,labels) in enumerate(train_loader):
        #print(batch.shape)
        #print(batch[0].shape)

        batch = batch.view(64,3,INPUT_DIM, INPUT_DIM)  # flatten input data   #此处直接将len(batch)=64
        output_data = torch.zeros(64, 3, 36, 36)
        output_data[:, :, 2:34, 2:34] = batch
        batch=output_data

        #print('training step = %d'%(count))
        count+=1

        if CUDA:
            batch = batch.cuda()

        batch_error = convrbm.contrastive_divergence(batch)

        epoch_error += batch_error
        torch.cuda.empty_cache()
        if count%100==0 and count!=0:
            break
    print('==================================================')
    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

    #print(torch.mean(convrbm.conv1_weights))
    #print(convrbm.conv1_weights)


torch.save(convrbm.conv1_weights, 'conv1_weights_fine.pth')
torch.save(convrbm.conv1_hidden_bias, 'conv1_hidden_bias_fine.pth')
print(torch.mean(convrbm.conv1_weights))

########## EXTRACT FEATURES ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
 if len(batch)!=16:
    #print(i)
    batch = batch.view(len(batch), 3,INPUT_DIM, INPUT_DIM)  # flatten input data
    output_data = torch.zeros(64, 3, 36, 36)
    output_data[:, :, 2:34, 2:34] = batch
    batch = output_data

    if CUDA:
        batch = batch.cuda()
    x=convrbm.sample_hidden(batch).cpu() #64*8*8
    train_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = x.view(64, -1).detach().numpy()
    train_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()
 if len(batch)==16:
     batch = batch.view(len(batch), 3, INPUT_DIM, INPUT_DIM)  # flatten input data
     output_data = torch.zeros(16, 3, 36, 36)
     output_data[:, :, 2:34, 2:34] = batch
     batch = output_data

     if CUDA:
         batch = batch.cuda()
     x = convrbm.sample_hidden(batch).cpu()  # 64*8*8
     train_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = x.view(16, -1).detach().numpy()
     train_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()


for i, (batch, labels) in enumerate(test_loader):
 if len(batch)!=16:
    #print(i)
    batch = batch.view(len(batch), 3,INPUT_DIM, INPUT_DIM)  # flatten input data
    output_data = torch.zeros(64, 3, 36, 36)
    output_data[:, :, 2:34, 2:34] = batch
    batch = output_data

    if CUDA:
        batch = batch.cuda()
    x=convrbm.sample_hidden(batch).cpu() #64*8*8
    test_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = x.view(64, -1).detach().numpy()
    test_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()
 if len(batch)==16:
     batch = batch.view(len(batch), 3, INPUT_DIM, INPUT_DIM)  # flatten input data
     output_data = torch.zeros(16, 3, 36, 36)
     output_data[:, :, 2:34, 2:34] = batch
     batch = output_data

     if CUDA:
         batch = batch.cuda()
     x = convrbm.sample_hidden(batch).cpu()  # 64*8*8
     test_features[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = x.view(16, -1).detach().numpy()
     test_labels[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = labels.numpy()

torch.save(train_features.reshape(50000,64,7,7),'train_features.pth')
torch.save(test_features.reshape(10000,64,7,7),'test_features.pth')
torch.save(train_labels,'train_labels.pth')
torch.save(test_labels,'test_labels.pth')


