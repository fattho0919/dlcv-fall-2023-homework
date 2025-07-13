from PIL import Image
import json
import clip
import random
import torch
import os
import csv
import sys

test_image_path = sys.argv[1]
id2label_path = sys.argv[2]
output_csv_path = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

output_list = []

#data

label = json.load(open(id2label_path, encoding="utf-8"))
label = list(label.values())
text_list = ["a photo of a " + label[i] for i in range(len(label))]

image_list = os.listdir(test_image_path)

#model
model, processor = clip.load("ViT-L/14")
model = model.to(device)
model.eval()


for i in range(len(image_list)):
    print(f"progress: {i}/{len(image_list)}", end="\r")
    image = Image.open(os.path.join(test_image_path, image_list[i]))
    img_inputs = processor(image).unsqueeze(0).to(device)

    text = clip.tokenize(text_list).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(img_inputs, text)
        probs = logits_per_image.softmax(dim=1)

    output_list.append({'filename':image_list[i], 'label':probs.argmax().item()})

fields = ["filename", "label"]
with open(output_csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for row in output_list:
        writer.writerow(row)
print("csv file saved")