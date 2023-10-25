from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision 
from torch.utils.data import DataLoader
import torch
import numpy as np
import time

from globals_var import * 
from unet import UNet
from diffusion import Diffusion

EVALUATE_2MODELS = True 
test = torchvision.datasets.Flowers102(
    root=".", download=True, split="val", transform=data_transforms
) 
dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

path_to_results = 'output'
if EVALUATE_2MODELS:
    small_model_bin = torch.load(f'{path_to_results}/model1.bin')
    small_model = UNet(64, [1, 2, 4, 8, 16], 1)
    small_model.load_state_dict(small_model_bin)
    pytorch_total_params = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
    print('small model trainable params', pytorch_total_params)
    big_model_bin = torch.load(f'{path_to_results}/model.bin')
    big_model = UNet(64, [1, 2, 4, 8, 16], 2)
    big_model.load_state_dict(big_model_bin)
    pytorch_total_params = sum(p.numel() for p in big_model.parameters() if p.requires_grad)
    print('big model trainable params', pytorch_total_params)

else: 
    test_model = torch.load(f'{path_to_results}/model1.bin')
    model_instance = UNet(UNET_CHANNEL_BASE, UNET_CHANNEL_MULT, UNET_DEPTH)
    model_instance.load_state_dict(test_model)

device = "cuda" if torch.cuda.is_available() else "cpu"

diffusion_instance = Diffusion(TIMESTEPS)
real_images = []
fake_images = []
times = []
# for _, batch in enumerate(dataloader):
batch = next(iter(dataloader))
batch = batch[0].to(device)
real_images.extend([reverse_transforms(img.cpu()) for img in batch])
start = time.time()
if EVALUATE_2MODELS:
    fake = diffusion_instance.p_sample_2models(big_model, small_model, batch.shape, TIMESTEPS)
    # fake = diffusion_instance.p_sample_2models(small_model,big_model, batch.shape, TIMESTEPS)
else: 
    fake = diffusion_instance.p_sample_loop(model_instance, batch.shape, TIMESTEPS)
end = time.time()
times.append((end-start)/BATCH_SIZE)
batch = next(iter(dataloader))
batch = batch[0].to(device)

fake_images.extend([reverse_transforms(img.cpu()) for img in fake[0]])

# batch = next(iter(dataloader))
# batch = batch[0].to(device)
# fake_images.extend([reverse_transforms(img.cpu()) for img in batch])

fake_images = torch.tensor(fake_images).permute(0, 3, 1, 2)
real_images = torch.tensor(real_images).permute(0, 3, 1, 2)
fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
print('average time of image generation', np.array(times).mean())