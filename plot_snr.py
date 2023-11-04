from diffusion import Diffusion
from matplotlib import pyplot as plt


timestep = 300
diffusion = Diffusion(timestep)
plt.figure(figsize=(10,6))
# plt.plot(range(timestep), [get_snr(i, diffusion) for i in range(timestep)])
plt.plot(range(timestep), diffusion.snr)
plt.title(f'T={timestep}')
plt.xlabel('Timestep')
plt.ylabel('Signal-noise ratio')
plt.yscale('log')
plt.axhline(y = 0.01, color = 'r', linestyle = '-') 
plt.axhline(y = 1, color = 'r', linestyle = '-') 
plt.savefig(f'output/snr.png')