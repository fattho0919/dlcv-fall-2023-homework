import random
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
from DANN import CNNModel
from PIL import Image
import sys
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 999078
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

image_path = sys.argv[1]
output_path = sys.argv[2]

class dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.data_path = data_path
        self.image_list = os.listdir(data_path)
        self.image_list.sort()

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.image_list[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, f"{self.image_list[index]}"
    
    def __len__(self):
        return len(self.image_list) 

batch_size = 32
target_val_dataset = dataset(data_path=image_path, transform=transforms.ToTensor())
target_val_dataloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
dann = CNNModel()
if "usps" in image_path.lower():
    dann.load_state_dict(torch.load('./hw2_checkpoint/DANN_usps.pth'))
    print("Load usps model")
elif "svhn" in image_path.lower():
    dann.load_state_dict(torch.load('./hw2_checkpoint/DANN_svhn.pth'))
    print("Load shvn model")
else:
    print("Cannot recognize the domain, choose usps model as default")
    dann.load_state_dict(torch.load('./hw2_checkpoint/DANN_usps.pth'))
    
dann.to(device)

# Validation
dann.eval()
output_class_list = []
with torch.no_grad():
    for i, (target_img, img_name) in enumerate(target_val_dataloader):
        target_img = target_img.to(device)

        class_output_target = dann(input_data=target_img, train=False)
        class_output_target = torch.argmax(class_output_target, dim=1)
        for i, j in zip(img_name, class_output_target):
            output_class_list.append({'image_name':i, 'label':j.item()})


fields = ["image_name", "label"]
with open(output_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for row in output_class_list:
        writer.writerow(row)
print("csv file saved")

    