import torch 

from torchvision import transforms

from matplotlib import pyplot as plt 
import matplotlib.animation as animation

import numpy as np 

import time 


from globals_var import * 
from unet import UNet
from diffusion import sample, init_forward_variables


path_to_results = 'output'
# load trained model 
test_model = torch.load(f'{path_to_results}/model.bin')
model_instance = UNet(UNET_CHANNEL_BASE, UNET_CHANNEL_MULT, UNET_DEPTH)
model_instance.load_state_dict(test_model)

reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
])

num_channels = 3
init_forward_variables(TIMESTEPS)

# generate image from noise 
samples = sample(model_instance, IMG_SIZE, 1, num_channels, TIMESTEPS)
# plt.imshow(reverse_transforms(samples[-1][0]))    # final image 

fig = plt.figure()
ims = []
for i in range(TIMESTEPS):
    im = plt.imshow(reverse_transforms(samples[i][0]), animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save(f'{path_to_results}/diffusion.gif')


