import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import os
from DDPMnDDIM import DDPM
import random
import numpy as np
from UNet import UNet
import imageio
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 999078
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

class DDIM_dataset(Dataset):
    def __init__(self, noise_path):
        self.noise_path = noise_path
        self.noise_list = os.listdir(noise_path)
        self.noise_list.sort()

    
    def __getitem__(self, index):
        noise = torch.load(os.path.join(self.noise_path, self.noise_list[index])).squeeze(0)
        return noise
    
    def __len__(self):
        return len(self.noise_list)

def DDIM_inference():
    n_T = 50
    n_sample = 1
    
    ddim_dataset = DDIM_dataset(noise_path=noise_path)
    ddim_dataloader = DataLoader(ddim_dataset, batch_size=1, shuffle=False)
    model = UNet()
    model.load_state_dict(torch.load(pretrained_model_path))
    model.to(device)

    ddim = DDPM(nn_model=model, betas=(1e-4, 2e-2), n_T=n_T, device=device, drop_prob=0.1, ddim=True, ddim_duf_st=1000)
    ddim.to(device)
    ddim.eval()

    with torch.no_grad():
        for i, noise in enumerate(tqdm(ddim_dataloader)):
            noise = noise.to(device)
            x_gen = ddim.ddim_sample(noise=noise, n_sample=n_sample, size=(1,3,256,256) , device=device, ddim_eta=0, guide_w=0)
            img = torch.clamp(x_gen, -1, 1)
            # x_gen = torch.clamp(x_gen, 0, 1)
            save_image(img, os.path.join(output_path, f"{i:02d}.png"), normalize=True)
    
            print(f"\nimage {i:02d}.png saved")
            print("-"*50)

if __name__ == "__main__":
    noise_path = sys.argv[1]
    output_path = sys.argv[2]
    pretrained_model_path = sys.argv[3]
    DDIM_inference()