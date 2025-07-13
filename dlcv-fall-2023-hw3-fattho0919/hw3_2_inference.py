import torch
import torchvision.transforms as transforms
from dataset import testing_dataset
from tokenizer import BPETokenizer
from original_decoder import Decoder_with_adapter, Config
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import clip
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

validation_image_path = sys.argv[1]
output_json_path = sys.argv[2]
cfg_path = sys.argv[3]

encoder_path = "./encoder.json"
vocab_path = "./vocab.bpe"

clip_model, clip_processor = clip.load("ViT-L/14@336px", device=device, jit=False)
clip_model = clip_model.to(device)
for param in clip_model.parameters():
    param.requires_grad = False

tokenizer = BPETokenizer(encoder_path, vocab_path)

cfg = Config(cfg_path)
decoder = Decoder_with_adapter(cfg)

# freeze the weights of the decoder but adapter and cross attention
for param in decoder.parameters():
    param.requires_grad = False
decoder = decoder.to(device)
print("Total parms", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

validation_dataset = testing_dataset(validation_image_path, tokenizer, clip_model, clip_processor)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
        
def validation(load_model=False):
    output_dict = {}
    encoded_caption = torch.tensor([50256]).unsqueeze(0)
    if load_model:
        decoder.load_state_dict(torch.load("./hw3_2_adapter_9429_7168.pth"), strict=False)
    decoder.eval()
    with torch.no_grad():
        validation_progress_bar = tqdm(validation_dataloader)
        for feature, image_name in validation_progress_bar:
            feature = feature.to(device)
            encoded_caption = encoded_caption.to(device)
            output = decoder.caption_image_beam_search(encoded_caption, feature)
            caption = output.tolist()
            caption = tokenizer.decode(caption[:-1])
            output_dict[image_name[0].split(".")[0]] = caption
        validation_progress_bar.close()
    with open(output_json_path, "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    validation(load_model=True)


        