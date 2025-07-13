import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

class image_caption_pair_dataset(Dataset):
    def __init__(self, image_path, caption_json_path, tokenizer, clip_model, clip_processor, transform=None):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.transform = transform
        with open(caption_json_path, "r") as f:
            self.caption_json = json.load(f)
        self.caption_data_list = self.caption_json["annotations"]
        self.image_data_list = self.caption_json["images"]
        self.caption_list = {}
        for item in self.caption_data_list:
            if item["image_id"] not in self.caption_list:
                self.caption_list[item["image_id"]] = [item["caption"]]
            else:
                self.caption_list[item["image_id"]].append(item["caption"])
        self.image_id_list = [item["id"] for item in self.image_data_list]
        self.image_list = {item["id"]: item["file_name"] for item in self.image_data_list}

        self.max_seq_len = 0

        for image_id in self.image_id_list:
            captions = self.caption_list[image_id]
            for i in range(len(captions)):
                captions[i] = self.tokenizer.encode(captions[i])
                if captions[i][-1] != 13:
                    captions[i].append(13)
                captions[i] = torch.tensor(captions[i])
                captions[i] = torch.cat([torch.tensor([50256]), captions[i], torch.tensor([50256])])
                if len(captions[i]) > self.max_seq_len:
                    self.max_seq_len = len(captions[i])

    def pad(self, image_id):
        captions = self.caption_list[image_id].copy()
        for i in range(len(captions)):
            padding_tokens = torch.tensor([50256]).repeat(self.max_seq_len - len(captions[i]))
            captions[i] = torch.cat([captions[i], padding_tokens])
        return captions
    
    def target_pad(self, image_id):
        captions = self.caption_list[image_id].copy()
        for i in range(len(captions)):
            padding_tokens = torch.tensor([-100]).repeat(self.max_seq_len - len(captions[i]) + 1)
            captions[i] = torch.cat([captions[i], padding_tokens])[1:]
        return captions

    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        image_id = self.image_id_list[idx]
        image_name = self.image_list[image_id]

        encoded_captions = self.pad(image_id)
        encoded_target_captions = self.target_pad(image_id)

        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = self.clip_processor(image).unsqueeze(0).to(device)
        image = self.clip_model.encode_image(image).to(device, dtype=torch.float32)
        return image, encoded_captions, encoded_target_captions, image_name

class testing_dataset(Dataset):
    def __init__(self, image_path, tokenizer, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.image_list = os.listdir(image_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]

        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        image = self.clip_processor(image).unsqueeze(0).to(device)
        image = self.clip_model.encode_image(image).to(device, dtype=torch.float32)
        return image, image_name