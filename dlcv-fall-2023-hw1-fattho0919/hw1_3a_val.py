import os
import torch
import torch.nn as nn
import torchvision.models as models
import imageio
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the VGG16-FCN32s model
class VGG16_FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_FCN32s, self).__init__()
        self.vgg = models.vgg16().features
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False),
        )
        
    def forward(self, x):
        x_size = x.size()
        x = self.vgg(x)
        x = self.fcn(x)
        x = nn.functional.interpolate(x, size=(x_size[-2], x_size[-1]), mode='bilinear', align_corners=False)

        return x

class CustomDataset(Dataset):
    def __init__(self, data_dir, image_transform=None, mask_transform=None):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = [filename for filename in os.listdir(self.data_dir) if filename.endswith('_sat.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.images[idx])
        mask_name = os.path.join(self.data_dir, self.images[idx].replace('_sat.jpg', '_mask.png'))
        
        image = imageio.v2.imread(img_name)
        mask = imageio.v2.imread(mask_name)

        if self.image_transform:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)

        if self.mask_transform:
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
    
        mask = np.array(mask)
        mask = (mask == 255).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        original_mask = copy.deepcopy(mask)

        mask[original_mask == 3] = 0  
        mask[original_mask == 6] = 1  
        mask[original_mask == 5] = 2  
        mask[original_mask == 2] = 3  
        mask[original_mask == 1] = 4  
        mask[original_mask == 7] = 5  
        mask[original_mask == 0] = 6  

        mask = torch.tensor(mask)

        return image, mask, self.images[idx]

def output_to_image(batch_preds, batch_names, out_path):
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]
        pred_img[np.where(pred == 1)] = [255, 255, 0]
        pred_img[np.where(pred == 2)] = [255, 0, 255]
        pred_img[np.where(pred == 3)] = [0, 255, 0]
        pred_img[np.where(pred == 4)] = [0, 0, 255]
        pred_img[np.where(pred == 5)] = [255, 255, 255]
        pred_img[np.where(pred == 6)] = [0, 0, 0]
        imageio.imwrite(os.path.join(
            out_path, name.replace('.jpg', '.png')), pred_img)

val_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

num_classes = 7  
batch_size = 20

validation_dataset = CustomDataset(data_dir='./hw1_data/p3_data/validation', image_transform=val_image_transform, mask_transform=None)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
model = VGG16_FCN32s(num_classes)
model.load_state_dict(torch.load('./checkpoint/hw1_3a.pth'))
model.to(device)
model.eval()
val_loss = 0.0
with torch.no_grad():
    for val_inputs, val_masks, filename in validation_loader:
        val_inputs = val_inputs.to(device)
        val_masks = val_masks.to(device)

        val_outputs = model(val_inputs)
        loss = criterion(val_outputs, val_masks)
        val_loss += loss.item()

        val_preds = torch.argmax(val_outputs, dim=1)
        output_to_image(val_preds, filename, './hw1_3a_pre_img')

print(f'Validation Loss: {val_loss / len(validation_loader):.4f}')
    