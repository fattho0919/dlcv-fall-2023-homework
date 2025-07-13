# import module
import os
import torch.nn as nn
import sys
from PIL import Image
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageDataset(Dataset):
    def __init__(self, csv_path:str ,data_path:str, transform=None) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.data_path = data_path
        self.transform = transform
        with open(self.csv_path, 'r') as f:
            self.image_files = [row['filename'] for row in csv.DictReader(f)]

    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, index) -> torch.Tensor:
        image_path = os.path.join(self.data_path, self.image_files[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[index]

def testing(model, dataloader, device):
    model.eval()
    data = []
    id = 0
    
    with torch.no_grad():
        for inputs, filename in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for f,l in zip(filename,predicted):
                data.append({'id':id, 'filename':f,'label':int(l)})
                id += 1
    return data


def main():
    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])
    ])

    validation_dataset = ImageDataset(csv_path=input_csv_path, data_path=testing_images_path, transform=valid_transform)
    validation_loader = DataLoader(validation_dataset, batch_size=5)

    model = models.resnet50()
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
    model.load_state_dict(torch.load('./checkpoint/hw1_2_inference.pth'))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    output_data = testing(model, validation_loader, device)
    
    fields = ["id", "filename", "label"]
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in output_data:
            writer.writerow(row)
    print("csv file saved")

if __name__ == '__main__':
    input_csv_path = sys.argv[1]
    testing_images_path = sys.argv[2]
    output_csv_path = sys.argv[3]
    main()