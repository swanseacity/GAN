import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image


LATENT_DIM = 100
LR = 0.0002
EPOCH = 100
BATCH_SIZE = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()
path = r'C:\Users\user\python\laboratory\GAN\data\MNIST\raw'
train_DS = datasets.MNIST(root = path,train=True, download=True, transform=transform)
test_DS = datasets.MNIST(root = path,train=False, download=True, transform=transform)

train_DL = torch.utils.data.DataLoader(train_DS, batch_size = BATCH_SIZE, shuffle = True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size = BATCH_SIZE, shuffle = True)

class GENERATOR(nn.Module):
    def __init__(self):
        super().__init__()
        
        def block(input_dim, output_dim, normalize = True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.ReLU())
            return layers
        
        self.model = nn.Sequential(
            *block(LATENT_DIM, 128, normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512, 1024),
            nn.Linear(1024,1*28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], 1, 28, 28)
        return img
    
class DISCRIMINATOR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(1*28*28, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        
    def forward(self, img):
        flattened = img.view(img.shape[0],-1)
        output = self.model(flattened)
        
        return output
    

G = GENERATOR()
D = DISCRIMINATOR()

G.to(device)
D.to(device)

#손실함수
def D_LOSS(D,x,z):
    return -(torch.log(D(x))+torch.log(1-D(G(z)))).mean()
def G_LOSS(G,D,z):
    return -torch.log(D(G(z))).mean()

#옵티마이저
opt_G = optim.Adam(G.parameters(),lr=LR,betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(),lr=LR,betas=(0.5, 0.999))

import time
sample_interval = 100
start_time = time.time()

for ep in range(EPOCH):
    for i, (x_batch,_) in enumerate(train_DL):
        x_batch = x_batch.to(device)
        #G learning
        opt_G.zero_grad()
        z = torch.normal(mean=0, std=1, size=(x_batch.shape[0], LATENT_DIM)).to(device)
        g_loss = G_LOSS(G,D,z)
        g_loss.backward()
        opt_G.step()
        #D learning
        opt_D.zero_grad()
        d_loss = D_LOSS(D,x_batch,z)
        d_loss.backward()
        opt_D.step()
        
        done = ep*len(train_DL)+i
        if done % sample_interval == 0:
            save_image(G(z)[:25],f'{done}.png',nrow=5, normalize=True)
    
    print(f'[Epoch {ep}/{EPOCH}] [D(X):{D(x_batch).mean()}] [D(G(z)):{D(G(z)).mean()}] [Elapsed time: {time.time()-start_time}]')
    

