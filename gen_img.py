import torch 

from torchvision import transforms

from matplotlib import pyplot as plt 
import matplotlib.animation as animation

import numpy as np 

import time 


from globals_var import * 
from unet import UNet
from diffusion import Diffusion, get_noise


path_to_results = 'output'
# load trained model 
test_model = torch.load(f'{path_to_results}/model.bin')
model_instance = UNet(UNET_CHANNEL_BASE, UNET_CHANNEL_MULT, UNET_DEPTH)
model_instance.load_state_dict(test_model)
diffusion_instance = Diffusion(TIMESTEPS)

num_channels = 3

# generate image from noise 
samples = diffusion_instance.sample(model_instance, IMG_SIZE, 1, num_channels, TIMESTEPS)
plt.imshow(reverse_transforms(samples[-1][0]))    # final image 

fig = plt.figure()
ims = []
for i in range(TIMESTEPS):
    im = plt.imshow(reverse_transforms(samples[i][0]), animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# animate.save(f'{path_to_results}/diffusion.gif')
plt.imsave(f'{path_to_results}/out.png', reverse_transforms(samples[-1][0]))

plt.figure(figsize=(10,6))
plt.hist(samples[-1][0].flatten())
plt.savefig(f'{path_to_results}/distribution.png')

plt.figure(figsize=(10,6))
plt.hist(get_noise(samples[-1][0].shape).flatten())
plt.savefig(f'{path_to_results}/noise_distribution.png')



