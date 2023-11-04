import torchvision 
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import sys
from globals_var import * 
from unet import UNet, ModelCombined
from diffusion import Diffusion, get_noise
from matplotlib import pyplot as plt 

test = torchvision.datasets.Flowers102(
    root=".", download=True, split="val", transform=data_transforms
) 
dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
path_to_results = 'output'
diffusion = Diffusion(TIMESTEPS)
device = "cuda" if torch.cuda.is_available() else "cpu"


fast_model_config = torch.load(f'{path_to_results}/model_fast.bin')
fast_model = UNet(64, [1, 2, 4, 8, 16], 1)
fast_model.load_state_dict(fast_model_config)
fast_model.to(device)

content_model_config = torch.load(f'{path_to_results}/model.bin')
content_model = UNet(64, [1, 2, 4, 8, 16], 2)
content_model.load_state_dict(content_model_config)
content_model.to(device)


batch = next(iter(dataloader))
batch = batch[0].to(device)

losses_content = []
losses_fast = []

with torch.no_grad():
    for i in range(TIMESTEPS):
        t = torch.full((batch.shape[0],), i, device=device).long()
        _, loss = diffusion.p_losses(content_model, batch, t)
        losses_content.append(float(loss))
        _, loss = diffusion.p_losses(fast_model, batch, t)
        losses_fast.append(float(loss))

plt.figure(figsize=(10,6))
plt.plot(range(TIMESTEPS), losses_content, label='depth=2')
plt.plot(range(TIMESTEPS), losses_fast, label='depth=1')
plt.legend()
plt.savefig(f'{path_to_results}/val_loss.png')
    
