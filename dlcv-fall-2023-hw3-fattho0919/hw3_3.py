import torch
import os
from PIL import Image
from tokenizer import BPETokenizer
from original_decoder import Decoder_with_adapter_visualization, Config
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image
import matplotlib.cm as cm
import json
import clip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg_path = "./hw3_data/p2_data/decoder_model.bin"
encoder_path = "./encoder.json"
vocab_path = "./vocab.bpe"
image_path = "./hw3_data/p2_data/images/val/"
checkpoints_path = "./checkpoints/"
output_json_path = "./hw3_2_result_adapter_visualization.json"

clip_model, clip_processor = clip.load("ViT-L/14@336px", device=device, jit=False)
clip_model = clip_model.to(device)
for param in clip_model.parameters():
    param.requires_grad = False
clip_model.eval()

tokenizer = BPETokenizer(encoder_path, vocab_path)
cfg = Config(cfg_path)
decoder = Decoder_with_adapter_visualization(cfg)
for param in decoder.parameters():
    param.requires_grad = False
decoder = decoder.to(device)

print("Total parms", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

image_list = ['000000179758.jpg', '461413605.jpg']
        
def validation(load_model=False):
    output_dict = {}
    encoded_caption = torch.tensor([50256]).unsqueeze(0)
    if load_model:
        decoder.load_state_dict(torch.load(f"hw3_2_adapter_9429_7168.pth"), strict=False)
    decoder.eval()
    with torch.no_grad():
        for image_name in image_list:
            original_image = Image.open(os.path.join(image_path, image_name))
            processed_image = clip_processor(original_image).unsqueeze(0).to(device)
            image = clip_model.encode_image(processed_image).to(device, dtype=torch.float32).unsqueeze(0)
            image = image.to(device)
            encoded_caption = encoded_caption.to(device)
            output, attention_map = decoder.weight_visualization(encoded_caption, image)

            summed_attention_map = [i.view(1, 12, 8, 8).sum(dim=1 ,keepdim=True) for i in attention_map]
            attention_map_normalized = [(i - i.min()) / (i.max() - i.min()) for i in summed_attention_map]
            resized_attention_map = [i.repeat_interleave(42, dim=2).repeat_interleave(42, dim=3) for i in attention_map_normalized]
            attention_map_np = [i.squeeze(0).squeeze(0).detach().numpy() for i in resized_attention_map]
            
            original_image_np = processed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            original_image_np = (original_image_np - original_image_np.min()) / (original_image_np.max() - original_image_np.min())
            for i in range(len(attention_map_np)):
                heatmap = plt.cm.rainbow(attention_map_np[i])[:, :, :3]
                overlayed_image = heatmap * 0.4 + original_image_np * 0.6
                final_image = Image.fromarray((overlayed_image * 255).astype(np.uint8))
                final_image.save(f"./{image_name}_{i}.png")
            caption = output.tolist()
            caption = tokenizer.decode(caption[:-1])
            output_dict[image_name.split(".")[0]] = caption

    with open(output_json_path, "w") as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    validation(load_model=True)


        