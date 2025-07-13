# import module
import os
import torch.nn as nn
import imageio
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageDataset(Dataset):
    def __init__(self, data_path:str, transform=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.transform = transform

        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith('.jpg')]

    def __len__(self) -> int:
        return len(self.image_files)
    
    def _get_label_from_filename(self, filename: str) -> int:
        return int(filename.split('_')[0])

    def __getitem__(self, index) -> torch.Tensor:
        image_path = os.path.join(self.data_path, self.image_files[index])
        image = imageio.v2.imread(image_path)
        label = self._get_label_from_filename(self.image_files[index])

        if self.transform:
            image = self.transform(image)

        return image, label

def validate(model, dataloader, criterion, device):
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
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    validation_dataset = ImageDataset(data_path="hw1_data/p2_data/office/val", transform=valid_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=20)

    model = models.resnet50()
    model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 65),
        )
    model.load_state_dict(torch.load('./checkpoint/hw2_2c_ResNet_33_model.pth'))
    
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    validation_loss, validation_accuracy = validate(model, validation_loader, criterion, device)
    print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")

if __name__ == '__main__':
    main()