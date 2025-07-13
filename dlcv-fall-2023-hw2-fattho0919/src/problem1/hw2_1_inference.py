import torch
from torchvision.utils import save_image, make_grid
import numpy as np
import os
from DDPMnDDIM import DDPM, ContextUnet
from digit_classifier import Classifier, DATA, load_checkpoint
import random
import numpy as np
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 999078
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

def accuracy_score(net, data_loader):
    correct = 0
    total = 0
    net.eval()
    print('===> start evaluation ...')
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = net(imgs)
            _, pred = torch.max(output, 1)
            correct += (pred == labels).detach().sum().item()
            total += len(pred)
    print('acc = {} (correct/total = {}/{})'.format(float(correct)/total, correct, total))
    
    return float(correct)/total 

def sample_image():
    n_T = 400
    n_classes = 10
    n_feat = 200 # 128 ok, 256 better (but slower)
    n_sample = 100
    checkpoint_path = './hw2_checkpoint/'

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load(checkpoint_path + "hw2_1.pth"))
    ddpm.to(device)

    digit_classifier = Classifier()
    path = "./Classifier.pth"
    load_checkpoint(path, digit_classifier)
    digit_classifier = digit_classifier.to(device)

    ddpm.eval()
    with torch.no_grad():
        for i in range(10):
            x_gen, x_gen_store, labels = ddpm.sample(n_sample=n_sample, size=(3, 28, 28),label=i, device=device, guide_w=2)
            for j, (img, label) in enumerate(zip(x_gen, labels)):
                save_image(img, os.path.join(output_path , f"{label}_{1+j%n_sample:03d}.png"))
            print(f'saved label {i} images at ' + output_path + "\n")
    
    val_data_loader = torch.utils.data.DataLoader(DATA(output_path), batch_size=2, num_workers=4, shuffle=False)
    accuracy_score(digit_classifier, val_data_loader)
    print("-"*50)

if __name__ == "__main__":
    output_path = sys.argv[1]
    sample_image()