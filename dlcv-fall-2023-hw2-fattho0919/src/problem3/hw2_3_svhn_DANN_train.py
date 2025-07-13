import random
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
from DANN import CNNModel
from PIL import Image
from tqdm import tqdm
import pandas as pd
from torchview import draw_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 999078
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# load data

source_img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

target_img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

lr = 1e-4
batch_size = 128
n_epoch = 100

source_train_dataset = dataset_mnist(csv_file_path='./hw2_data/digits/mnistm/train.csv', data_path='./hw2_data/digits/mnistm/data', transform=source_img_transform)
target_train_dataset = dataset_mnist(csv_file_path='./hw2_data/digits/svhn/train.csv', data_path='./hw2_data/digits/svhn/data', transform=target_img_transform)
target_val_dataset = dataset_mnist(csv_file_path='./hw2_data/digits/svhn/val.csv', data_path='./hw2_data/digits/svhn/data', transform=transforms.ToTensor())

source_train_dataloader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
target_train_dataloader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
target_val_dataloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

min_len_dataloader = min(len(source_train_dataloader), len(target_train_dataloader))

dann = CNNModel()
dann.to(device)
optimizer = optim.Adam(dann.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss()

best_acc = 0.0   #0.4341

# training
for ep in range(n_epoch):
    print(f'epoch {ep} / {n_epoch}')
    dann.train()
    combined_dataloader = zip(source_train_dataloader, target_train_dataloader)
    progress_bar = tqdm(enumerate(combined_dataloader, start=0), total=min_len_dataloader)

    for i, ((source_img, source_class), (target_img, _)) in progress_bar:
        optimizer.zero_grad()  # Reset gradients to zero at the beginning of each iteration
        
        p = float(i + ep * min_len_dataloader) / n_epoch / min_len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        alpha = torch.tensor(alpha).to(device)

        # Process source data
        domain_label_source = torch.zeros(len(source_class), dtype=torch.int64).to(device)

        source_img = source_img.to(device)
        source_class = source_class.to(device)
        
        class_output_source, domain_output_source = dann(input_data=source_img, alpha=alpha)
        class_loss_source = loss(class_output_source, source_class)
        domain_loss_source = loss(domain_output_source, domain_label_source)

        # Process target data
        domain_label_target = torch.ones(len(target_img), dtype=torch.int64).to(device)
        
        target_img = target_img.to(device)
        
        _, domain_output_target = dann(input_data=target_img, alpha=alpha)
        domain_loss_target = loss(domain_output_target, domain_label_target)

        # Calculate total loss, perform backpropagation and optimizer step
        total_loss = class_loss_source + domain_loss_source + domain_loss_target
        total_loss.backward()
        optimizer.step()

        # Update progress bar description
        progress_bar.set_description(
            f'Training: '
            f'Training Loss: {total_loss.item():.4f}, '
            f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
        )
    
    # Validation
    acc = 0
    progress_bar = tqdm(enumerate(target_val_dataloader, start=0), total=len(target_val_dataloader))
    dann.eval()
    with torch.no_grad():
        for i, (target_img, target_class) in enumerate(target_val_dataloader):

            target_img = target_img.to(device)
            target_class = target_class.to(device)

            class_output_target = dann(input_data=target_img, train=False)
            class_loss_target = loss(class_output_target, target_class)
            acc = acc + torch.sum(torch.argmax(class_output_target, dim=1) == target_class).item()

            progress_bar.set_description(
                f'Validation: '
                f'Validation Loss: {class_loss_target.item():.4f} '
                f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}'
            )
    acc = acc / (len(target_val_dataloader)*batch_size)
    
    if acc > best_acc:
        best_acc = acc
        torch.save(dann.state_dict(), './hw2_checkpoint/DANN_svhn.pth')
        print(f'best acc = {best_acc:.4f}')
        print('saved model at ' + './hw2_checkpoint/DANN_svhn.pth')
    print(f"Best acc: {best_acc:.4f}")
    print("-"*50)
    