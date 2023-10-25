import torchvision
from torchvision import transforms 

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np 
import json
import os 

from unet import UNet
from diffusion import Diffusion
from globals_var import * 
import globals_var


def save_config_to_json(path, config_module):
    module_dict = {key: value for key, value in globals_var.__dict__.items() if (not key.startswith('_')) and (type(value) in [int, list])}
    with open(f'{path}/config.json', 'w') as fp:
        json.dump(module_dict, fp)

path_to_results = "output"
os.makedirs(path_to_results, exist_ok=True)
save_config_to_json(path_to_results, globals_var)


diffusion_instance = Diffusion(TIMESTEPS)
train = torchvision.datasets.Flowers102(
    root=".", download=True, split="test", transform=data_transforms
)  # train and test are switched
test = torchvision.datasets.Flowers102(
    root=".", download=True, split="train", transform=data_transforms
) 
data = torch.utils.data.ConcatDataset([train, test])
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = UNet(UNET_CHANNEL_BASE, UNET_CHANNEL_MULT, UNET_DEPTH)
model.to(device)

optimizer = Adam(model.parameters(), lr=2e-4)

epochs_results = {}
for epoch in range(EPOCHS):
    print('epoch', epoch)
    epoch_timesteps = []
    epoch_loss = []
    for step, batch in enumerate(dataloader):

        optimizer.zero_grad()
        batch = batch[0].to(device)
        batch_size = batch.shape[0]
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()
        epoch_timesteps.append(t.tolist())
        loss_full, loss = diffusion_instance.p_losses(model, batch, t)
        loss_per_img = [img_loss.mean().tolist() for img_loss in loss_full]
        epoch_loss.append(loss_per_img)

        if step == 0:
            print(f"Loss for step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()
    epochs_results[epoch] = {'timesteps':epoch_timesteps, 'losses':epoch_loss}
    with open(f'{path_to_results}/epochs.json', 'w') as fp:
        json.dump(epochs_results, fp)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'{path_to_results}/model.bin')
        print('model saved')

