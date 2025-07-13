import torch
from byol_pytorch import BYOL
from torchvision import models
from torchvision import transforms
import imageio
import os
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDataset(Dataset):
    def __init__(self, data_path:str, transform=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.transform = transform

        self.image_files = [f for f in os.listdir(self.data_path) if f.endswith('.jpg')]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index) -> torch.Tensor:
        image_path = os.path.join(self.data_path, self.image_files[index])
        image = imageio.v2.imread(image_path)

        if self.transform:
            image = self.transform(image)

        return image

training_dataset = ImageDataset('hw1_data/p2_data/mini/train', transform=transforms.ToTensor())
training_dataloader = DataLoader(training_dataset, batch_size=100, shuffle=True)

resnet = models.resnet50(weights=None)

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool',
    use_momentum = False     
)

learner = learner.to(device)
optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
total_iterations = len(training_dataloader)
best_loss = float('inf')

for epoch in range(1000):
    pretrained_loss = 0
    for images in training_dataloader:
        images = images.to(device)
        loss = learner(images)
        pretrained_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch: {epoch} average loss: {pretrained_loss / total_iterations}')

    if (pretrained_loss / total_iterations) < best_loss:
        torch.save(resnet.state_dict(), './checkpoint/hw1_2ssl_backbone.pth')
        best_loss = pretrained_loss / total_iterations
        print(f"epoch = {epoch}, model saved!")
        print(f'best loss: {best_loss}')
    print('-' * 50)
