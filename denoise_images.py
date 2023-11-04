import torch 
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision

from matplotlib import pyplot as plt 

import numpy as np 

import time 

from globals_var import *
from unet import UNet
from diffusion import Diffusion, get_noise


def test_denoising(batch, model, timestep, diffusion):
    t = torch.tensor([timestep])
    img_noise = diffusion.q_sample(batch, t)
    img_denoised = diffusion.p_sample_loop_denoise(img_noise, model, timestep)
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(reverse_transforms(img_noise[0]))
    axes[1].imshow(reverse_transforms(img_denoised[0]))
    axes[2].imshow(reverse_transforms(batch[0]))
    plt.savefig(f'{path_to_results}/denoised_img{timestep}.png')

path_to_results = 'output'
# load trained model 
test_model = torch.load(f'{path_to_results}/model_fast.bin')
model_instance = UNet(64, [1, 2, 4, 8, 16], 1)
model_instance.load_state_dict(test_model)
diffusion = Diffusion(TIMESTEPS)

num_channels = 3

test = torchvision.datasets.Flowers102(
    root=".", download=True, split="val", transform=data_transforms
) 
dataloader = DataLoader(test, batch_size=1, shuffle=True, drop_last=True)

batch = next(iter(dataloader))
batch = batch[0]

plt.axis('off')
num_images = 10
stepsize = int(TIMESTEPS/num_images)

fig, axes = plt.subplots(2, num_images, figsize=(30, 5))


for idx in reversed(range(TIMESTEPS)):
    if idx % stepsize == 0:
        t = torch.Tensor([idx]).type(torch.int64)
        forward_sample = diffusion.q_sample(batch, t, noise=None, clipping=False)
        img = diffusion.p_sample_loop_denoise(forward_sample, model_instance, idx, clipping=True)
        plot_id = int(idx // stepsize)
        axes[0][plot_id].imshow(reverse_transforms(img[0].cpu()))
        axes[1][plot_id].imshow(reverse_transforms(forward_sample[0].cpu()))
plt.savefig(f'{path_to_results}/forward_backward.png')
    