from PIL import Image
import json
from transformers import CLIPProcessor, CLIPModel
import random
import torch
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_top5_predictions(image, image_filename, probs, labels, topk=5):
    top_probs, top_labels = torch.topk(probs, topk)
    top_probs = top_probs.detach().cpu().numpy().flatten()
    top_labels = ["a photo of a " + labels[label] for label in top_labels.detach().cpu().numpy().flatten()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    ax1.imshow(image)
    ax1.axis('off')  # Hide the axis
    ax1.set_title(f'Original Image: {image_filename}')

    # Plot the top-5 predictions
    ax2.barh(np.arange(topk), top_probs, color='skyblue')
    ax2.set_yticks(np.arange(topk))
    ax2.set_yticklabels(top_labels)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Probabilities')
    ax2.set_title('Top-5 Predictions')

    plt.tight_layout()
    plt.savefig(f'./{image_filename}.png')


output_list = []
acc = 0

#data
id2label_path = "./hw3_data/p1_data/id2label.json"
test_image_path = "./hw3_data/p1_data/val/"
output_csv_path = "./p1_val.csv"

label = json.load(open(id2label_path, encoding="utf-8"))
label = list(label.values())
text_list = ["a photo of a " + label[i] for i in range(len(label))]

# text_list = ["No " + label[i] for i in range(len(label))]
# for i in range(len(text_list)):
#     text_list[i] = text_list[i] + ", no score."

image_list = os.listdir(test_image_path)

#model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = model.to(device)
model.eval()

# csv file
for i in range(len(image_list)):
    print(f"progress: {i}/{len(image_list)}", end="\r")
    image = Image.open(os.path.join(test_image_path, image_list[i]))
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    output_list.append({'filename':image_list[i], 'label':probs.argmax().item()})

    if probs.argmax().item() == int(image_list[i].split("_")[0]):
        acc += 1

print(f"accuracy: {acc/len(image_list)}")

fields = ["filename", "label"]
with open(output_csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    for row in output_list:
        writer.writerow(row)
print("csv file saved")

# plot
sampled_images = random.sample(image_list, 3)

for image_filename in sampled_images:
    image = Image.open(os.path.join(test_image_path, image_filename))
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Call the plotting function
    plot_top5_predictions(image, image_filename, probs[0], label)