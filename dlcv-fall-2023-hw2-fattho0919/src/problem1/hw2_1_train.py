import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os
import pandas as pd
from DDPMnDDIM import DDPM, ContextUnet
from PIL import Image
from digit_classifier import Classifier, DATA, load_checkpoint
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 999078
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

def worker_init_fn(worker_id):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)

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

class dataset_mnist(Dataset):
    def __init__(self, csv_file_path, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.file = pd.read_csv(csv_file_path)
        self.image_list = self.file['image_name']
        self.label_list = self.file['label']

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.image_list[index])
        img = Image.open(img_path).convert('RGB')
        label = self.label_list[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.image_list)    

def train_mnist():
    n_epoch = 30
    batch_size = 256
    n_T = 400
    n_classes = 10
    n_feat = 200   #128~256
    lrate = 1e-4
    n_sample = 100
    output_path = './hw2_1_output/'
    checkpoint_path = './hw2_checkpoint/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    digit_classifier = Classifier()
    path = "./Classifier.pth"
    load_checkpoint(path, digit_classifier)
    digit_classifier = digit_classifier.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epoch, eta_min=0.0)

    train_dataset = dataset_mnist(csv_file_path='./hw2_data/digits/mnistm/train.csv', data_path='./hw2_data/digits/mnistm/data', transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    best_acc = 0.0

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()
        pbar = tqdm(train_dataloader)
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            pbar.set_description(f"loss: {loss:.4f}")
            optim.step()

        if ep<5:
            continue
        ddpm.eval()
        with torch.no_grad():
            for i in range(10):
                x_gen, x_gen_store, labels = ddpm.sample(n_sample, (3, 28, 28),i, device, guide_w=2)
                for j, (img, label) in enumerate(zip(x_gen, labels)):
                    save_image(img, output_path + f"{label}_{1+j%n_sample:03d}.png")
                print(f'saved label {i} images at ' + output_path + "\n")

        val_data_loader = torch.utils.data.DataLoader(DATA(output_path), batch_size=2, num_workers=4, shuffle=False)
        acc = accuracy_score(digit_classifier, val_data_loader)

        # optionally save model
        if acc > best_acc:
            best_acc = acc
            torch.save(ddpm.state_dict(), checkpoint_path + f"hw2_1.pth")
            print('saved model at ' + checkpoint_path + f"hw2_1.pth")

        scheduler.step()
        print(f'best acc = {best_acc:.4f}')
        print("-"*50)

if __name__ == "__main__":
    train_mnist()