import torch
import os
from PIL import Image
import numpy as np
from p2_evaluate import main
from argparse import Namespace
import torchvision.transforms as transforms
from dataset import image_caption_pair_dataset
from tokenizer import BPETokenizer
from original_decoder import Decoder_with_prefix, Config
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import json
from tqdm import tqdm
import clip
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg_path = "./hw3_data/p2_data/decoder_model.bin"
encoder_path = "./encoder.json"
vocab_path = "./vocab.bpe"
training_image_path = "./hw3_data/p2_data/images/train/"
training_caption_json_path = "./hw3_data/p2_data/train_preprocessed.json"
validation_image_path = "./hw3_data/p2_data/images/val/"
validation_caption_json_path = "./hw3_data/p2_data/val.json"
checkpoints_path = "./checkpoints/"
output_json_path = "./hw3_2_result_prefix.json"

clip_model, clip_processor = clip.load("ViT-L/14@336px", device=device, jit=False)
clip_model = clip_model.to(device)
for param in clip_model.parameters():
    param.requires_grad = False

tokenizer = BPETokenizer(encoder_path, vocab_path)

n_prefix = 20
cfg = Config(cfg_path)
decoder = Decoder_with_prefix(cfg, n_prefix)
decoder = decoder.to(device)
print("Total parms", sum(p.numel() for p in decoder.parameters() if p.requires_grad))

epochs = 10
train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])
train_dataset = image_caption_pair_dataset(training_image_path, training_caption_json_path, tokenizer, clip_model, clip_processor, device, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_dataset = image_caption_pair_dataset(validation_image_path, validation_caption_json_path, tokenizer, clip_model, device, clip_processor)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

def train():
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=len(train_dataloader)/2, gamma=0.8)
    best_score = 0

    for epoch in range(epochs):
        decoder.train()
        print(f"Epoch {epoch + 1} / {epochs}")
        train_progress_bar = tqdm(train_dataloader)
        training_loss = 0.0
        for image, encoded_captions, encoded_target_captions, _ in train_progress_bar:
            image = image.to(device)
            
            for encoded_caption, encoded_target_caption in zip(encoded_captions, encoded_target_captions):
                encoded_caption = encoded_caption.to(device)
                encoded_target_caption = encoded_target_caption.to(device)
                output = decoder(encoded_caption, image)
                loss = criterion(output.reshape(-1, output.shape[-1])[n_prefix:], encoded_target_caption.reshape(-1))
            # o = torch.argmax(output, dim=-1).tolist()[0]
            # if 50256 in o:
            #     o = o[:o.index(50256)]
            # print(tokenizer.decode(o))
            # print(tokenizer.decode(encoded_target_caption[encoded_target_caption!=-100].tolist()))
            # print("-"*50)
            
            training_loss += loss.item()
            train_progress_bar.set_postfix({
                    'training_loss': training_loss / (train_progress_bar.n + 1),
                    })
            train_progress_bar.update()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        train_progress_bar.close()

        validation(load_model=False)
        args = Namespace(pred_file="/home/hongfa/DLCV/dlcv-fall-2023-hw3-fattho0919/hw3_2_result_prefix.json", images_root="/home/hongfa/DLCV/dlcv-fall-2023-hw3-fattho0919/hw3_data/p2_data/images/val", annotation_file="/home/hongfa/DLCV/dlcv-fall-2023-hw3-fattho0919/hw3_data/p2_data/val.json")
        cider_score, clip_score = main(args)

        if cider_score > best_score:
            best_score = cider_score
            trainable_weights = [name for name, param in decoder.named_parameters() if param.requires_grad == True]
            save_weights = {k: v for k, v in decoder.state_dict().items() if k in trainable_weights}
            torch.save(save_weights, checkpoints_path + f"hw3_2_prefix.pth")
            print("Saving model with highest cider score")
        # with torch.no_grad():
        #     validation_progress_bar = tqdm(validation_dataloader)
        #     validation_loss = 0.0
        #     for image, encoded_caption, encoded_target_caption, _ in validation_progress_bar:
        #         image = image.to(device)
        #         for encoded_caption, encoded_target_caption in zip(encoded_captions, encoded_target_captions):
        #             encoded_caption = encoded_caption.to(device)
        #             encoded_target_caption = encoded_target_caption.to(device)
        #             output = decoder(encoded_caption, image)
        #             loss = criterion(output.reshape(-1, output.shape[-1]), encoded_target_caption.reshape(-1))

        #         validation_loss += loss.item()
        #         validation_progress_bar.set_postfix({
        #                 'validation_loss': validation_loss / (validation_progress_bar.n + 1),
        #                 })
        #         validation_progress_bar.update()
        #     validation_progress_bar.close()
        # print(f"average validation loss: {validation_loss / len(validation_dataloader)}")

        # if validation_loss < best_val_loss:
        #     best_val_loss = validation_loss
        #     torch.save(decoder.state_dict(), checkpoints_path + f"hw3_2.pth")
        #     print("Saving model with lowest validation loss")
        
def validation(load_model=False):
    output_dict = {}
    encoded_caption = torch.tensor([50256]).unsqueeze(0)
    if load_model:
        decoder.load_state_dict(torch.load(checkpoints_path + f"hw3_2_prefix.pth"), strict=False)
    decoder.eval()
    with torch.no_grad():
        validation_progress_bar = tqdm(validation_dataloader)
        for feature, _, _, image_name in validation_progress_bar:
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
    train()


        