# import module
import os
import torch.nn as nn
from PIL import Image
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
        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith('.png')]
        self.image_files = sorted(self.image_files, key=self.custom_sort)

    def custom_sort(self, item):
        parts = item.split('_')
        return (int(parts[0]), int(parts[1].split('.')[0]))

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

        return image, label, self.image_files[index]

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, filename in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total

def warmup_and_decay_lambda(epoch):
    warmup_epochs = 30
    total_epochs = 100 
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs)

def main():
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])
    ])

    validation_dataset = ImageDataset(data_path="hw1_data/p1_data/val_50", transform=valid_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=20)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 50),
    )
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('./checkpoint/hw1_1b.pth'))
    model = model.to(device)

    validation_loss, validation_accuracy = validate(model, validation_loader, criterion, device)
    print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")

if __name__ == '__main__':
    main()