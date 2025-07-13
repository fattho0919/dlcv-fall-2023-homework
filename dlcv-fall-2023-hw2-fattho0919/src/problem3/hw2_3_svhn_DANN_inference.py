import random
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch
from DANN import CNNModel
from PIL import Image
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 999078
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def extract_features(model, source_dataloader, target_dataloader, device):
    model.eval()
    features = []
    class_labels = []
    domain_labels = []

    with torch.no_grad():
        for inputs, lbls in source_dataloader:
            inputs = inputs.to(device)
            feats = model.feature(inputs).flatten(start_dim=1)
            features.append(feats.cpu().numpy())
            class_labels.append(lbls)
            for i in range(len(lbls)):
                domain_labels.append(0)
        
        for inputs, lbls in target_dataloader:
            inputs = inputs.to(device)
            feats = model.feature(inputs).flatten(start_dim=1)
            features.append(feats.cpu().numpy())
            class_labels.append(lbls)
            for i in range(len(lbls)):
                domain_labels.append(1)

    features = np.vstack(features)
    class_labels = np.hstack(class_labels)
    domain_labels = np.hstack(domain_labels)
    return features, class_labels, domain_labels
    
def visualize_tsne(features, labels, save_path):

    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            color=colors[i],
            label=label
        )
    
    plt.legend(loc='best')
    plt.title('t-SNE of svhn DANN Features')
    plt.savefig(save_path)

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
batch_size = 32

source_val_dataset = dataset_mnist(csv_file_path='./hw2_data/digits/mnistm/val.csv', data_path='./hw2_data/digits/mnistm/data', transform=transforms.ToTensor())
source_val_dataloader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
target_val_dataset = dataset_mnist(csv_file_path='./hw2_data/digits/svhn/val.csv', data_path='./hw2_data/digits/svhn/data', transform=transforms.ToTensor())
target_val_dataloader = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
dann = CNNModel()
dann.load_state_dict(torch.load('./hw2_checkpoint/DANN_svhn.pth'))
dann.to(device)

# Validation
acc = 0
dann.eval()
with torch.no_grad():
    for i, (target_img, target_class) in enumerate(target_val_dataloader):
        target_img = target_img.to(device)
        target_class = target_class.to(device)

        class_output_target = dann(input_data=target_img, train=False)
        acc = acc + torch.sum(torch.argmax(class_output_target, dim=1) == target_class).item()

acc = acc / (len(target_val_dataloader)*batch_size)

features, class_labels, domain_labels = extract_features(dann, source_val_dataloader, target_val_dataloader, device)
visualize_tsne(features, class_labels, './DANN_svhn_tsne_class.png')
visualize_tsne(features, domain_labels, './DANN_svhn_tsne_domain.png')

print(f'Validation Accuracy: {acc:.4f}')

    