from torchvision import transforms
import numpy as np 

KERNEL_SIZE = 3 
BATCH_SIZE = 128
# image size has to be % 2^ number_of_downscaling == 0 
IMG_SIZE = 64
TIMESTEPS = 300
UNET_CHANNEL_BASE = 64
UNET_CHANNEL_MULT = [1,2,4,8,16]
UNET_DEPTH = 1
EPOCHS = 100

data_transforms = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(), # tensor with values [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1), # Scale between [-1, 1] 
])

reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
])