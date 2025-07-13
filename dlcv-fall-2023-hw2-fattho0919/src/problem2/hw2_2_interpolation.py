import torch
from torchvision.utils import save_image
import numpy as np
import os
from DDPMnDDIM import DDPM
import random
import numpy as np
from UNet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 999078
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

def slerp(x1, x2, alpha):
    theta = torch.acos(torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2)))
    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * x1
        + torch.sin(alpha * theta) / torch.sin(theta) * x2
    )

def DDIM_inference():
    n_T = 50
    n_sample = 1
    output_path = './hw2_2_output/'
    pretrained_model_path = './hw2_data/face/'
    noise_path = "./hw2_data/face/noise/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    noise_0 = torch.load(os.path.join(noise_path, "00.pt"))
    noise_1 = torch.load(os.path.join(noise_path, "01.pt"))
    
    model = UNet()
    model.load_state_dict(torch.load(pretrained_model_path + "UNet.pt"))
    model.to(device)

    ddim = DDPM(nn_model=model, betas=(1e-4, 2e-2), n_T=n_T, device=device, drop_prob=0.1, ddim=True, ddim_duf_st=1000)
    ddim.to(device)
    ddim.eval()

    noise_0 = noise_0.to(device)
    noise_1 = noise_1.to(device)

    with torch.no_grad():
        for alpha in torch.arange(0.0, 1.01, 0.1):
            alpha = alpha.to(device)
            noise = slerp(noise_0, noise_1, alpha)
            x_gen = ddim.ddim_sample(noise=noise, n_sample=n_sample, size=(1,3,256,256) , device=device, ddim_eta=0, guide_w=0).squeeze(0)
            img = torch.clamp(x_gen, min=-1, max=1)
            save_image(img, output_path + f"alpha{alpha:1f}.png", normalize=True)
            print(f"alpha{alpha:2f}.png saved")
            
if __name__ == "__main__":
    DDIM_inference()