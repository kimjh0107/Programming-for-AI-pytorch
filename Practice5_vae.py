# %% 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init 
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
import matplotlib as mpl
from IPython.display import Image 
import time
import copy
# set hyperparamters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10

# download mnist dataset
mnist_train = dset.MNIST('./', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])

# set dataloader 
dataloaders = {}
dataloaders['train']  = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
dataloaders['val'] = DataLoader(mnist_val, batch_size=batch_size ,shuffle=False)
dataloaders['test'] = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# len(dataloaders["train"])
# len(dataloaders["train"].dataset)

# Model 
class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 256), nn.Tanh(),)
        
        self.fc_mu = nn.Linear(256,10)
        self.fc_var = nn.Linear(256,10)

        self.decoder = nn.Sequential(nn.Linear(10,256), nn.Tanh(), nn.Linear(256, 28*28), nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self,z): # x 가 아닌 새로운 값 z
        recon = self.decoder(z)
        return recon
    
    def forward(self, x):
        batch_size = x.size(0) # x: (batch_size, 1, 28, 28), numpy.size() function count the number of elements along a given axis.
        mu, log_var = self.encode(x.view(batch_size, -1)) #view() helps to get a new view of array with the same data
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var


# device & loss func & Optimizer 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BCE = torch.nn.BCELoss(reduction = 'sum')
# vae 의 경우 자체적인 loss func을 통해서 새롭게 정의해줘야됨 
def loss_func(x, recon_x, mu, log_var):
    BCE_loss = BCE(recon_x, x.view(-1, 28*28))
    KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE_loss + KLD_loss
    # pow() function returns the value of x to the power of y (xy)
    # exp() function in Python allows users to calculate the exponential value

model = VariationalAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



# train process 
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    since = time.time()

    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 100000000


    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # each epoch train / val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Iterate over data 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)

                # zero the parameters gradients 
                optimizer.zero_grad()

                # forward 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, mu, log_var = model(inputs)
                    loss = criterion(inputs, outputs, mu, log_var)

                    # backward 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics 
                running_loss += loss.item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'train':
                train_loss_history.append(epoch_loss)

            elif phase == 'val':
                val_loss_history.append(epoch_loss)

            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


        print()   

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history

best_model, train_loss_history, val_loss_history = train_model(model, dataloaders, loss_func, optimizer, num_epochs=num_epochs)
# %%
plt.plot(train_loss_history, label='train')
plt.plot(val_loss_history, label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

