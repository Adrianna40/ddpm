import torch
import torch.nn.functional as F
import time 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np 

class Diffusion:
    def __init__(self, timesteps):
        self.timesteps = timesteps 
        self.betas = get_linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod
        [:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.tensor(np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ))
        self.snr = 1.0 / (1 - self.alphas_cumprod) - 1

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None: 
            noise = get_noise(x_start.shape).to(x_start.device)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        noised_img = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        noised_img = torch.clip(noised_img, min=-1.0, max=1.0) 
        return noised_img

    # backward diffusion loss 
    def p_losses(self, denoise_model, x_start, t):
        device = next(denoise_model.parameters()).device
        noise = get_noise(x_start.shape).to(device)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = denoise_model(x_noisy, t)
        # l1_loss # huber_loss # kl_div 
        loss_full = F.l1_loss(noise, predicted_noise, reduction='none')  # saving loss for each image separately, for analysis of different timesteps 
        loss = F.l1_loss(noise, predicted_noise)
        return loss_full, loss

    # source https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=_s-Al2lJ2c8T
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        device = next(model.parameters()).device
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            # posterior_log_variance_t = extract(self.posterior_log_variance_clipped, t, x.shape)
    
            noise = get_noise(x.shape).to(device)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, model, shape, timesteps):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = get_noise(shape).to(device)
        imgs = []

        start = time.time()
        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
    
            img = torch.clip(img, min=-1.0, max=1.0)
            imgs.append(img.cpu())
        end = time.time()
        print('image generation took', end-start)
        return imgs
    
    @torch.no_grad()
    def p_sample_loop_denoise(self, noisy_img, model, timestep):
        device = next(model.parameters()).device
        shape = noisy_img.shape
        b = shape[0]

        for i in tqdm(reversed(range(0, timestep)), desc='sampling loop time step', total=timestep):
            noisy_img = self.p_sample(model, noisy_img, torch.full((b,), i, device=device, dtype=torch.long), i)
            noisy_img = torch.clip(noisy_img, min=-1.0, max=1.0)
      
        return noisy_img

    @torch.no_grad()
    def sample(self, model, image_size, batch_size, channels, timesteps):
        return self.p_sample_loop(model, (batch_size, channels, image_size, image_size), timesteps)
    
    @torch.no_grad()
    def p_sample_2models(self, big_model, small_model, shape, timesteps):
        device = next(big_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = get_noise(shape).to(device)
        imgs = []

        start = time.time()
        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            if i < 200 or i >= 600: 
                model = small_model 
            else:
                model = big_model
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            img = torch.clip(img, min=-1.0, max=1.0) 
            imgs.append(img.cpu())
        end = time.time()
        print('image generation took', end-start)
        return imgs


def get_linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def get_noise(shape):
    noise = torch.normal(mean=0.0, std=1.0, size=shape)

    # Replace values outside of the range [-1, 1]
    #noise = torch.where((noise < -1) | (noise > 1), torch.rand(shape) * 2 - 1, noise)

    return noise

    