from torchmetrics.image.fid import FrechetInceptionDistance
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
fake_images = []
times = []

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
    # model = ModelCombined(content_model, fast_model, diffusion)
    model = ModelCombined(fast_model, content_model, diffusion) # reversed order 

elif sys.argv[1]=='fast_model': 
    model_config = torch.load(f'{path_to_results}/model_fast.bin')
    model = UNet(64, [1, 2, 4, 8, 16], 1)
    model.load_state_dict(model_config)

elif sys.argv[1]=='content_model': 
    model_config = torch.load(f'{path_to_results}/model.bin')
    model = UNet(64, [1, 2, 4, 8, 16], 2)
    model.load_state_dict(model_config)

batch = next(iter(dataloader))
batch = batch[0].to(device)
real_images = [reverse_transforms(img.cpu()) for img in batch]

if sys.argv[1]=='fid_base':
    """
    comparing real images with real images to see what is the best possible FID score for this dataset 
    """
    fid_scores = []
    for _ in range(10):
        batch2 = next(iter(dataloader))
        batch2 = batch2[0].to(device)
        fake_images = [reverse_transforms(img.cpu()) for img in batch2]
        real_images = [reverse_transforms(img.cpu()) for img in batch]
        fake_images = torch.tensor(np.array(fake_images)).permute(0, 3, 1, 2)
        real_images = torch.tensor(np.array(real_images)).permute(0, 3, 1, 2)
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
        fid_score = float(fid.compute())
        fid_scores.append(fid_score)
        print(f"FID: {fid_score}")
        batch = next(iter(dataloader))
        batch = batch[0].to(device)
    print('average fid', np.array(fid_scores).mean())
    print('std fid', np.array(fid_scores).std())

else:
    times = []
    fid_scores = []
    for _ in range(10):
        model.cuda()
        start = time.time()
        t = torch.tensor([TIMESTEPS-1], device=device)
        img_noise = diffusion.q_sample(batch, t)
        fake = diffusion.p_sample_loop_denoise(img_noise, model, TIMESTEPS, clipping=True)
        end = time.time()
        fake_images = [reverse_transforms(img.cpu()) for img in fake]
        time_generation = (end-start)/BATCH_SIZE
        times.append(time_generation)
        print('average time of image generation', time_generation)
        fake_images = torch.tensor(np.array(fake_images)).permute(0, 3, 1, 2)
        real_images = torch.tensor(np.array(real_images)).permute(0, 3, 1, 2)
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)
        fid_score = float(fid.compute())
        fid_scores.append(fid_score)
        print(f"FID: {fid_score}")
        batch = next(iter(dataloader))
        batch = batch[0].to(device)
        real_images = [reverse_transforms(img.cpu()) for img in batch]

    print('average fid', np.array(fid_scores).mean())
    print('std fid', np.array(fid_scores).std())
    print('average time', np.array(times).mean())

