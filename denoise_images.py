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
test_model = torch.load(f'{path_to_results}/model1.bin')
model_instance = UNet(UNET_CHANNEL_BASE, UNET_CHANNEL_MULT, UNET_DEPTH)
model_instance.load_state_dict(test_model)
diffusion_instance = Diffusion(TIMESTEPS)

num_channels = 3

test = torchvision.datasets.Flowers102(
    root=".", download=True, split="val", transform=data_transforms
) 
dataloader = DataLoader(test, batch_size=1, shuffle=True, drop_last=True)

batch = next(iter(dataloader))
batch = batch[0]

test_denoising(batch, model_instance, 50, diffusion_instance)
test_denoising(batch, model_instance, 100, diffusion_instance)
test_denoising(batch, model_instance, 200, diffusion_instance)
test_denoising(batch, model_instance, 500, diffusion_instance)
test_denoising(batch, model_instance, 999, diffusion_instance)