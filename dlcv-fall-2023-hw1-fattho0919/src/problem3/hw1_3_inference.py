import os
import torch
import torchvision.models as models
import imageio
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import sys
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

testing_images_path = sys.argv[1]
output_images_path = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        image = imageio.v2.imread(img_name)

        if self.image_transform:
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_transform(image)


        return image, self.images[idx]

def output_to_image(batch_preds, batch_names, out_path):
    for pred, name in zip(batch_preds, batch_names):
        output_name = name.replace('_sat.jpg', '_mask.png')
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]
        pred_img[np.where(pred == 1)] = [255, 255, 0]
        pred_img[np.where(pred == 2)] = [255, 0, 255]
        pred_img[np.where(pred == 3)] = [0, 255, 0]
        pred_img[np.where(pred == 4)] = [0, 0, 255]
        pred_img[np.where(pred == 5)] = [255, 255, 255]
        pred_img[np.where(pred == 6)] = [0, 0, 0]
        imageio.imwrite(os.path.join(out_path, output_name), pred_img)

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

num_classes = 7  
batch_size = 5

testing_dataset = CustomDataset(data_dir=testing_images_path, image_transform=image_transform, mask_transform=None)
testing_loader = DataLoader(testing_dataset, batch_size=batch_size)

model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
model.aux_classifier = models.segmentation.deeplabv3.FCNHead(1024, num_classes)
model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
model.load_state_dict(torch.load('./checkpoint/hw1_3_inference.pth'))
model.to(device)
model.eval()

with torch.no_grad():
    for inputs, filename in testing_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)['out']

        preds = torch.argmax(outputs, dim=1)
        output_to_image(preds, filename, output_images_path)

print("Mask images saved to: " + output_images_path)