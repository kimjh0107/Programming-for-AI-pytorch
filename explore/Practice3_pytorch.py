##############################################################
#################### 1. PyTorch and Numpy ####################
##############################################################

# %% 
import torch 
import numpy as np
# %%
np_array_1 = np.array([1, 2, 3, 4])
np_array_2 = np.array([5, 6, 7, 8])
torch_tensor_1 = torch.tensor([1, 2, 3, 4])
torch_tensor_2 = torch.tensor([5 ,6 ,7, 8])

print (np_array_1)
print (np_array_2)
print (torch_tensor_1)
print (torch_tensor_2)

# %%
#### Same operations with identical grammer ####
# numpy 
print(np_array_1.shape)

# torch 
print(torch_tensor_1.shape)
print(torch_tensor_2.size) # size and shpe operation is identical in torch 

# %%
#### Concatenate ####
# numpy 
np_concate = np.concatenate([np_array_1, np_array_2], axis = 0)
print(np_concate.shape)

# torch 
torch_concate = torch.cat([torch_tensor_1, torch_tensor_2], dim = 0)
print(torch_concate.shape)

# %%
#### reshape ####
# numpy
np_reshaped = np.concate.reshape(4,2)
print(np_reshaped.shape)

torch_reshaped = torch_concate.view(4,2)
print(torch_reshaped.shape)

# %%
#### manipulation tensors ####
x = np.array([1,2,3])
x_repeat = x. repeat(2) # array([1, 1, 2, 2, 3, 3])

x = torch.tensor([1, 2, 3])
x_repeat = x.repeat(2)

x_repeat = x.view(3, 1).repeat(1, 2).view(-1)
print (x_repeat.shape) # torch.Size([6])

x_repeat = x.view(3, 1).repeat(1, 2)
print (x_repeat.shape) # torch.Size([3, 2])

# %%
# similar manipulation operation: stack & repeat
x = torch.tensor([1, 2, 3]) # torch.Size([3])
x_repeat = x.repeat(4) # torch.Size([12])
x_stack = torch.stack([x, x, x, x]) # torch.Size([4, 3])



# %%
##############################################################
######  2. Tensor operations under GPU utilization ###########
##############################################################
a = torch.ones(3)
b = torch.randn(100, 50, 3) 

c = a + b # shape (100,50,3)
# %%
# requires_grad = gradient 계산 True로 설정할시 
x = torch.ones(2,2,requires_grad = True)
print(x)

# retain_grad() -> gradient를 계산하지 않고 진행한다는 의미 
# 중간 지점에서 stored the grad할 경우에 사용 


# %%
##############################################################
######  4. nn.Module ###########
##############################################################
import torch.nn as nn 

X = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
linear_fn = nn.Linear(3,1)
# WX + b # in features = 3, out_feature = 1

Y = linear_fn(X) 
print(Y.shape) # torch.Size([2, 1])
print(Y.sum())

"""Other types of nn example 
nn.Conv2d
nn.RNNCell
nn.LSTMCell
nn.GRUCell
nn.Transformer"""

# %%
# How to design a customized model example
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Model, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.linear_2 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x 





##############################################################
# 5. MNIST classification with PyTorch (Logistic regression & MLP)
##############################################################
# S -> softmax -> cross-Entropy Loss 
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./', train=False, transform=transforms.ToTensor())
# 6만개, 1만개 

# Data loader
# mini batch size -> 왜 여기서 batch size를 train, test와 다르게 진행을 하는건지?? 
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)
# %%
# Define model class 
class Multinomial_logistic_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super(Multinomial_logistic_regression, self).__init__()
        self.fc = nn.Linear(input_size, output_size) 
        
    def forward(self, x):
        out = self.fc(x)
        return out

# Generate model 
model = Multinomial_logistic_regression(784, 10) # init(784,10)
# input dim = 184, output dim = 10
# input channel -> 1, 28, 28  (one channel, 28 pixel, 28 pixel)
model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.05, momentum =0.9)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

# Train model 
# Loss 
loss_fn = nn.CrossEntropyLoss()
# Train the model 
total_step = len(train_loader)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader): # mini batch for loop
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward() # automatic gradien calculation (autograd)
        optimizer.step() #update model parameters with requries_grad = True 

        if (i+1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
            .format(epoch+1, 10, i+1, total_step, loss.item()))


# %%
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
# %%
