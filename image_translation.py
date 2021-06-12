from __future__ import print_function
import argparse
import os
import random
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setting the seed
manualSeed = 999
print("THe value of the seed is" + str(manualSeed))
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# Root directory for dataset
dataroot = "datasets/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

#Device to run on 
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1
fake_label = 0


#Creating dataset for the dataloader
def prepare_dataset():
    
    trans = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

    dataset = dset.ImageFolder(root=dataroot,
                               transform=trans )
    
    #Creating the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    

    #Plotting some images
    next_batch = next(iter(dataloader))

    '''plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(next_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()'''

    return dataloader



def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif classname.find('BatchNorm')  != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    """Class which implements the generator"""

    def __init__(self, ngpu):
        super(Generator, self).__init__()

        self.ngpu = ngpu

        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4,4 ,2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False ),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    

    def forward(self, X):
        return self.model(X)

class Discriminator(nn.Module):
    """Class which implements the discriminator"""

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()

        self.ngpu = ngpu

        self.model =   nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        return self.model(input)


def train_models(netG, netD, dataloader):
    """
        Train the gneerator and discriminator models. 
    """
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))  

    for epoch in range(num_epochs):
        # Going through the dataset completely is one epoch 

        for i, data in enumerate(dataloader, 0):
            
            netD.zero_grad()

            #Format batch
            real_device = data[0].to(device)

            b_size = real_device.size(0)

            y = torch.full((b_size, ), real_label, dtype=torch.float, device=device)

            #Doing a forward pass now
            y_hat = netD(real_device).view(-1)

            errD_real = criterion(y_hat, y)

            #Calculating the gradients for D in a backward pass
            errD_real.backward()
            D_x = y_hat.mean().item()

            # Training with a fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)

            # Generating fake images
            fake = netG(noise)

            #Populating he fake labels
            y.fill_(fake_label)

            # Classify all fake batch with D
            y_hat = netD(fake.detach()).view(-1)

            #COmputing the error
            errD_fake = criterion(y_hat, y)

            #Calculating the gradients for this batch accumulated with the previous gradients
            errD_fake.backward()
            D_G_z1 = y_hat.mean().item()

            #Computing the net error as sum of errors over the real and fake batches
            errD = errD_real + errD_fake

            #Update D
            optimizerD.step()

            # Updating the G network 
            netG.zero_grad()
            # TODO: Why is fake labels real for a generator cost? 
            y.fill_(real_label)

            y_hat = netD(fake).view(-1)

            errG = criterion(y_hat, y)

            # Getting the graients for G
            errG.backward()
            D_G_z2 = y_hat.mean().item()

            #Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            

            #Save errors for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            #PLotting some images
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and i == len(dataloader) - 1):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            iters += 1
        
        return img_list, D_losses, G_losses


if __name__ == '__main__':
    dataloader = prepare_dataset()
    netG = Generator(ngpu=ngpu).to(device=device)

    #Initializing he weights
    netG.apply(weights_init)

    netD = Discriminator(ngpu=ngpu).to(device=device)

    netD.apply(weights_init)


    print("The generator is: ")
    print(netG)

    print("The discriminator is: ")
    print(netD)
    

    img_list, D_losses, G_losses = train_models(netG, netD, dataloader)
    plt.figure()
    plt.figure("Generator and Discriminator loss duing training")

    plt.plot(G_losses, label='G')
    plt.plot(D_losses, label='D')

    plt.xlabel("iterations")
    plt.ylabel("losses")

    plt.legend()
    plt.show()


    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    #HTML(ani.to_jshtml())

    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
    plt.show()

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()


