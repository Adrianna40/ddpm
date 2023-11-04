import torch 

from torchvision import transforms

from matplotlib import pyplot as plt 
import matplotlib.animation as animation

import numpy as np 

import time 
import sys 

from globals_var import * 
from unet import UNet, ModelCombined
from diffusion import Diffusion, get_noise


path_to_results = 'output'
diffusion_instance = Diffusion(TIMESTEPS)

# load trained model 
if sys.argv[1]=='2models':
    fast_model_config = torch.load(f'{path_to_results}/model_fast.bin')
    fast_model = UNet(64, [1, 2, 4, 8, 16], 1)
    fast_model.load_state_dict(fast_model_config)
    fast_model_params_count = sum(p.numel() for p in fast_model.parameters() if p.requires_grad)
    print('small model trainable params', fast_model_params_count)
    content_model_config = torch.load(f'{path_to_results}/model.bin')
    content_model = UNet(64, [1, 2, 4, 8, 16], 2)
    content_model.load_state_dict(content_model_config)
    content_model_params_count = sum(p.numel() for p in content_model.parameters() if p.requires_grad)
    print('big model trainable params', content_model_params_count)
    model = ModelCombined(content_model, fast_model, diffusion_instance)
    # model = ModelCombined(fast_model, content_model, diffusion_instance) # reversed order 

elif sys.argv[1]=='fast_model': 
    model_config = torch.load(f'{path_to_results}/model_fast.bin')
    model = UNet(64, [1, 2, 4, 8, 16], 1)
    model.load_state_dict(model_config)

elif sys.argv[1]=='content_model': 
    model_config = torch.load(f'{path_to_results}/model.bin')
    model = UNet(64, [1, 2, 4, 8, 16], 2)
    model.load_state_dict(model_config)



num_channels = 3

# generate image from noise 

shape = (1, num_channels, IMG_SIZE, IMG_SIZE)
samples = diffusion_instance.p_sample_loop(model, shape, TIMESTEPS, img_noisy=None, clipping=False)

plt.imsave(f'{path_to_results}/out.png', reverse_transforms(samples[-1][0]))



