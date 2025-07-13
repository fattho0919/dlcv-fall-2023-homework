# import module
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn as nn
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, data_path:str, transform=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.transform = transform

        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith('.png')]

    def __len__(self) -> int:
        return len(self.image_files)
    
    def _get_label_from_filename(self, filename: str) -> int:
        return int(filename.split('_')[0])

    def __getitem__(self, index) -> torch.Tensor:
        image_path = os.path.join(self.data_path, self.image_files[index])
        image = Image.open(image_path).convert('RGB')
        label = self._get_label_from_filename(self.image_files[index])

        if self.transform:
            image = self.transform(image)

        return image, label

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            feats = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(inputs))))))))).squeeze()
            features.append(feats.cpu().numpy())
            labels.append(lbls)

    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels

def visualize_pca(features, labels, save_path):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = labels == label
        plt.scatter(
            principalComponents[indices, 0],
            principalComponents[indices, 1],
            color=colors[i],
        )
    
    plt.legend(loc='best')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of ResNet Features')
    
    # 保存圖片為文件
    plt.savefig(save_path)

def visualize_tsne(features, labels, save_path):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
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
        )
    
    plt.legend(loc='best')
    plt.title('t-SNE of ResNet Features')
    
    plt.savefig(save_path)

def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])
    ])

    testing_dataset = ImageDataset(data_path="hw1_data/p1_data/val_50", transform=transform)
    testing_dataloader = DataLoader(testing_dataset, batch_size=20)

    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 50),
    )
    model.load_state_dict(torch.load('./checkpoint/hw1_a.pth'))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    testing_loss, testing_accuracy = test(model, testing_dataloader, criterion, device)
    print(f"Testing Loss: {testing_loss:.4f}, Testing Accuracy: {testing_accuracy:.2f}%")

    features, labels = extract_features(model, testing_dataloader, device)
    visualize_pca(features, labels , './pca.png')

    epochs = [0, 49, 99] # 假設有這三個時期的模型
    for epoch in epochs:
        model_path = f'./checkpoint/hw1_a_{epoch}.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        model = model.to(device)

        testing_loss, testing_accuracy = test(model, testing_dataloader, criterion, device)
        print(f"epoch{epoch} Testing Loss: {testing_loss:.4f}, Testing Accuracy: {testing_accuracy:.2f}%")

        features, labels = extract_features(model, testing_dataloader, device)
        visualize_tsne(features, labels , f'./tsne_{epoch}.png')


if __name__ == '__main__':
    main()