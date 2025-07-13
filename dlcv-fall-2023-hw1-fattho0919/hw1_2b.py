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
    
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100 * correct / total

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

def warmup_and_decay_lambda(epoch):
    warmup_epochs = 30
    total_epochs = 100 
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs)

def main():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])
    ])

    training_dataset = ImageDataset(data_path="hw1_data/p2_data/office/train", transform=train_transform)
    validation_dataset = ImageDataset(data_path="hw1_data/p2_data/office/val", transform=valid_transform)

    training_loader = DataLoader(training_dataset, batch_size=20, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=20)

    model = models.resnet50(weights=None)
    model.load_state_dict(torch.load('./hw1_data/p2_data/pretrain_model_SL.pt'))
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1000),
        nn.BatchNorm1d(1000),  # Add BatchNormalization after the first linear layer
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1000, 500),
        nn.BatchNorm1d(500),  # Add BatchNormalization after the second linear layer
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(500, 65),
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_and_decay_lambda)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 100
    best_val_loss = float('inf')
    best_accuracy = 0.0
    train_writer = SummaryWriter(log_dir='runs/hw1_2b/train')
    validation_writer = SummaryWriter(log_dir='runs/hw1_2b/validation')

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, training_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = validate(model, validation_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%")

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), f'./checkpoint/hw1_2b.pth')
            print("The model's weight has been saved.")

        train_writer.add_scalar('Loss', train_loss, epoch)
        validation_writer.add_scalar('Loss', validation_loss, epoch)

        scheduler.step()
        print("-" * 50)
    
    print("Training is finished.")
    print(f"The best validation loss is {best_val_loss:.4f}, and the best validation accuracy is {best_accuracy:.2f}%")
    train_writer.close()
    validation_writer.close()
    


if __name__ == '__main__':
    main()