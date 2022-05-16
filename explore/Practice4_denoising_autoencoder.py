# %% 
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.init as init 
import torchvision.datasets as dset 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
import time 
import copy 

# %%
# download data 
mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])

# set hyperparamters 
batch_size = 256 
learning_rate = 0.0002
num_epochs = 10 

dataloaders = {}
dataloaders['train'] = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
dataloaders['val'] = DataLoader(mnist_val, batch_size = batch_size, shuffle=False)
dataloaders['test'] = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100,30),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(30,100),
            nn.ReLU(),
            nn.Linear(100, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        batch_size = x.size(0) # 최대한 모든 batch 끌어모으기 위해서 
        x = x.view(-1, 28*28) # reshape to -dimensional vector 
        encoded = self.encoder(x) # hidden vector -> encoders 
        out = self.decoder(encoded).view(batch_size, 1, 28, 28) # final out은 처음 input size와 동일하게 
        return out, encoded 

# set device 
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# set model 
model = Autoencoder().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



# %%
######### train denoising autoencoder -> 기존의 autoencoder와는 다른점은 noise를 추가해주는 것 외에는 동일함 
def train_model_D(model, dataloaders, criterion, optimizer, num_epochs=10):
    since = time.time()

    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 100000000

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)

        # train, valid if 문으로 선택하게
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # Iterate -> for문안에 for 문 
            for inputs, labels in dataloaders[phase]:
                noise = torch.zeros(inputs.size(0), 1, 28, 28) # noise 설정 
                nn.init.normal_(noise, 0, 0.1) # 정규분포에서의 값을 input값으로 
                noise = noise.to(device)
                inputs = inputs.to(device)
                noise_inputs = inputs + noise 

                # zero paramter gradients 
                optimizer.zero_grad()

                # forward 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, encoded = model(noise_inputs)
                    loss = criterion(outputs, inputs) # calculate a loss 

                    # train model만 backward, optimize 진행 (valid, test는 no)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics 
                running_loss += loss.item() * inputs.size(0) # + loss.item() 함수를 통해서 loss의 값을 가져오는 것 
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model -> 나중에 best model을 저장하기 위한 process
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            elif phase == 'val':
                val_loss_history.append(epoch_loss)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict()) # best model weight 저장 

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:.4f}'.format(best_val_loss))

    # load best model weights 
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history
    
"""
나중에 best model만 따로 추출하는 것도 좋긴 하지만, 다음에는 logger를 통해서 해당 파라미터도 저장하고 
best 모델 같은 경우도 이제 따로 model 파일을 만들어서 저장하도록 하는 틀을 만들어서 적용해보도록 하기 
"""
best_model_D, train_loss_history_D, val_loss_history_D = train_model_D(model, dataloaders, loss_func, optimizer, num_epochs=num_epochs)

# %%
plt.plot(train_loss_history_D, label = 'train') # label은 그냥 내가 설정 -> train이니 train으로 
plt.plot(val_loss_history_D, label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# %%
## test set에 대한 결과 확인 ## 
with torch.no_grad(): # torch.no_grad() 블록을 사용하는 이유는 해당 블록을 history 트래킹 하지 않겠다는 뜻이다.
    running_loss = 0.0
    for inputs, labels in dataloaders['test']:
        noise = nn.init.normal_(torch.FloatTensor(inputs.size(0), 1, 28, 28), 0, 0.1) # floattensor -> tensor 만들어줌 
        noise = noise.to(device)
        inputs = inputs.to(device)
        noise_inputs = inputs + noise

        outputs, encoded = best_model_D(noise_inputs)
        test_loss = loss_func(outputs, inputs)

        running_loss += test_loss.item() * inputs.size(0)
    
    test_loss = running_loss / len(dataloaders['test'].dataset)
    print(test_loss)


# %%
# denoise 를 통한 이미지 확인 -> 배경의 nosie는 깔끔해졌지만 epoch가 적어서 그런지 아직은 좀 흐린 부분 확인 (서버에서 epoch 늘려서 확인해보기)
out_img = torch.squeeze(outputs.cpu().data)
print(out_img.size()) # torch.Size([16, 28, 28])

for i in range(5):
    plt.subplot(1,2,1)
    plt.imshow(torch.squeeze(noise_inputs[i]).cpu().numpy(),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.show()


# %%
# tsne plot을 통해서 결과 한번 확인 
np.random.seed(42)
from sklearn.manifold import TSNE

test_dataset_array = mnist_test.data.numpy() / 255  # -> normalization
test_dataset_array = np.float32(test_dataset_array) # dtype -> float32, 사실 이 과정은 생략해도 크게 상관없음 
labels = mnist_test.targets.numpy()

test_dataset_array = torch.tensor(test_dataset_array)
inputs = test_dataset_array.to(device)
outputs, encoded = best_model_D(inputs)

encoded = encoded.cpu().detach().numpy()
tsne = TSNE()
X_test_2D = tsne.fit_transform(encoded)
X_test_2D = (X_test_2D - X_test_2D.min()) / (X_test_2D.max() - X_test_2D.min())



plt.figure(figsize=(10, 8))
cmap = plt.cm.tab10
plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=labels, s=10, cmap=cmap)
image_positions = np.array([[1., 1.]])
for index, position in enumerate(X_test_2D):
    dist = np.sum((position - image_positions) ** 2, axis=1)
    if np.min(dist) > 0.02: # if far enough from other images
        image_positions = np.r_[image_positions, [position]]
        imagebox = mpl.offsetbox.AnnotationBbox(
            mpl.offsetbox.OffsetImage(torch.squeeze(inputs).cpu().numpy()[index], cmap="binary"),
            position, bboxprops={"edgecolor": cmap(labels[index]), "lw": 2})
        plt.gca().add_artist(imagebox)
plt.axis("off")
plt.show()

# %%
