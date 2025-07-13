import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
    
class Attention_with_lora(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=16)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=16)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, cfg.n_embd)  # for queries
        self.feature_attn = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)  # for keys and values
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, feature, attention_map=False):
        B, T, C = x.size()  # batch, context, embedding
        _, S, _ = feature.size()  # batch, sequence length, embedding
        q = self.c_attn(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # query
        k, v = self.feature_attn(feature).split(self.n_embd, dim=2)  # key, value
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // self.n_head))
        att = att.masked_fill(self.bias[:, :, :T, :S] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        output = self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
        if attention_map:
            return output, q[:,:,-1,:]
        else:
            return output

class CrossAttention_with_lora(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, cfg.n_embd, r=16)  # for queries
        self.feature_attn = lora.Linear(cfg.n_embd, 2 * cfg.n_embd, r=16)  # for keys and values
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=16)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x, feature):
        B, T, C = x.size()  # batch, context, embedding
        _, S, _ = feature.size()  # batch, sequence length, embedding
        q = self.c_attn(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # query
        k, v = self.feature_attn(feature).split(self.n_embd, dim=2)  # key, value
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // self.n_head))
        att = att.masked_fill(self.bias[:, :, :T, :S] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Adapter(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.down_project = nn.Linear(cfg.n_embd, (cfg.n_embd // 2)-30)
        self.up_project = nn.Linear((cfg.n_embd // 2)-30, cfg.n_embd)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x + self.up_project(self.activation(self.down_project(x)))
        x = self.dropout(x)
        x = x + self.up_project(self.activation(self.down_project(x)))
        return x

class Block(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Block_with_cross_attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, x, feature=None):
        x = x + self.attn(self.ln_1(x))
        if feature is not None:
            x = x + self.cross_attn(self.ln_1(x), feature)
        x = x + self.mlp(self.ln_2(x))
        return x

class Block_with_adapter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
        self.adpt = Adapter(cfg)

    def forward(self, x, feature=None):
        x = x + self.attn(self.ln_1(x))
        if feature is not None:
            x = x + self.cross_attn(self.ln_1(x), feature)
        x = x + self.adpt(self.mlp(self.ln_2(x)))
        x = x + self.mlp(self.ln_2(x))
        return x

class Block_with_adapter_visualization(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))
        self.adpt = Adapter(cfg)

    def forward(self, x, feature=None):
        x = x + self.attn(self.ln_1(x))
        if feature is not None:
            o, visualize_weight = self.cross_attn(self.ln_1(x), feature, attention_map=True)
            x = x + o
        x = x + self.adpt(self.mlp(self.ln_2(x)))
        x = x + self.mlp(self.ln_2(x))
        return x, visualize_weight

class Block_with_lora(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention_with_lora(cfg)
        self.cross_attn = CrossAttention_with_lora(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, x, feature=None):
        x = x + self.attn(self.ln_1(x))
        if feature is not None:
            x = x + self.cross_attn(self.ln_1(x), feature)
        x = x + self.mlp(self.ln_2(x))
        return x

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        x = self.lm_head(self.transformer.ln_f(self.transformer.h(x)))
        return x

class Decoder_with_cross_attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block_with_cross_attention(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            print(f"loading decoder's checkpoint {self.cfg.checkpoint}")
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, feature: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        x = self.transformer.wte(x)
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x, feature)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x

    def caption_image(self, encoded_caption, feature, max_len=57):
        x = self.transformer.wte(encoded_caption)
        result_caption = encoded_caption

        for i in range(max_len):
            pos = torch.arange(0, x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
            x_pos = self.transformer.wpe(pos)
            x = x + x_pos
            for block in self.transformer.h:
                x = block(x, feature)
            x = self.transformer.ln_f(x)
            x = self.lm_head(x)
            x = x[:, -1:, :]
            o = torch.argmax(x, dim=-1)

            if o == 50256: # <eos>
                break

            result_caption = torch.cat((result_caption, o), dim=1)
            x = self.transformer.wte(result_caption)

        return result_caption[0, len(encoded_caption):]  # Return only the generated part
    
    def caption_image_beam_search(self, encoded_caption, feature, max_len=57, beam_width=5):
        # 初始化 Beam Search
        beams = [(encoded_caption, 0)]  # (caption, score)
        
        for step in range(max_len):
            new_beams = []
            for caption, score in beams:
                # 相同的 Transformer 操作
                x = self.transformer.wte(caption)
                pos = torch.arange(0, x.size()[1], device=x.device).unsqueeze(0)
                x = x + self.transformer.wpe(pos)
                for block in self.transformer.h:
                    x = block(x, feature)
                x = self.transformer.ln_f(x)
                x = self.lm_head(x)
                x = x[:, -1, :]

                # 選擇 top-k 候選
                softmax_scores = torch.softmax(x, dim=-1)
                topk_scores, topk_indices = torch.topk(softmax_scores, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # 添加一個額外的維度
                    next_score = score + topk_scores[0, i].log()  # 累計分數
                    new_caption = torch.cat((caption, next_token), dim=1)

                    if next_token == 50256:  # <eos>
                        return new_caption[0, len(encoded_caption):]
                    
                    new_beams.append((new_caption, next_score))

            # 保留分數最高的 beam_width 個候選
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # 選擇最佳的候選
        return beams[0][0][0, len(encoded_caption):]
    
class Decoder_with_adapter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block_with_adapter(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            print(f"loading decoder's checkpoint {self.cfg.checkpoint}")
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, feature: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        x = self.transformer.wte(x)
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x, feature)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x

    def caption_image(self, encoded_caption, feature, max_len=57):
        x = self.transformer.wte(encoded_caption)
        result_caption = encoded_caption

        for i in range(max_len):
            pos = torch.arange(0, x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
            x_pos = self.transformer.wpe(pos)
            x = x + x_pos
            for block in self.transformer.h:
                x = block(x, feature)
            x = self.transformer.ln_f(x)
            x = self.lm_head(x)
            x = x[:, -1:, :]
            o = torch.argmax(x, dim=-1)

            if o == 50256: # <eos>
                break

            result_caption = torch.cat((result_caption, o), dim=1)
            x = self.transformer.wte(result_caption)

        return result_caption[0, len(encoded_caption):]  # Return only the generated part
    
    def caption_image_beam_search(self, encoded_caption, feature, max_len=57, beam_width=5):
        # 初始化 Beam Search
        beams = [(encoded_caption, 0)]  # (caption, score)
        
        for step in range(max_len):
            new_beams = []
            for caption, score in beams:
                # 相同的 Transformer 操作
                x = self.transformer.wte(caption)
                pos = torch.arange(0, x.size()[1], device=x.device).unsqueeze(0)
                x = x + self.transformer.wpe(pos)
                for block in self.transformer.h:
                    x = block(x, feature)
                x = self.transformer.ln_f(x)
                x = self.lm_head(x)
                x = x[:, -1, :]

                # 選擇 top-k 候選
                softmax_scores = torch.softmax(x, dim=-1)
                topk_scores, topk_indices = torch.topk(softmax_scores, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # 添加一個額外的維度
                    next_score = score + topk_scores[0, i].log()  # 累計分數
                    new_caption = torch.cat((caption, next_token), dim=1)

                    if next_token == 50256:  # <eos>
                        return new_caption[0, len(encoded_caption):]
                    
                    new_beams.append((new_caption, next_score))

            # 保留分數最高的 beam_width 個候選
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # 選擇最佳的候選
        return beams[0][0][0, len(encoded_caption):]
    
class Decoder_with_adapter_visualization(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block_with_adapter_visualization(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            print(f"loading decoder's checkpoint {self.cfg.checkpoint}")
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def weight_visualization(self, encoded_caption, feature, max_len=57, beam_width=5):
        # 初始化 Beam Search
        beams = [(encoded_caption, 0)]  # (caption, score)
        visualize_weight_list = [([],0),([],0),([],0),([],0),([],0)]
        
        for step in range(max_len):
            new_beams = []
            for i, (caption, score) in enumerate(beams):
                a = []
                # 相同的 Transformer 操作
                x = self.transformer.wte(caption)
                pos = torch.arange(0, x.size()[1], device=x.device).unsqueeze(0)
                x = x + self.transformer.wpe(pos)
                for block in self.transformer.h:
                    x, visualize_weight = block(x, feature)
                    a.append(visualize_weight)
                visualize_weight = a[-1]
                x = self.transformer.ln_f(x)
                x = self.lm_head(x)
                x = x[:, -1, :]
                visualize_weight_list[i][0].append(visualize_weight.cpu())

                # 選擇 top-k 候選
                softmax_scores = torch.softmax(x, dim=-1)
                topk_scores, topk_indices = torch.topk(softmax_scores, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # 添加一個額外的維度
                    next_score = score + topk_scores[0, i].log()  # 累計分數
                    new_caption = torch.cat((caption, next_token), dim=1)

                    if next_token == 50256:  # <eos>
                        return new_caption[0, len(encoded_caption):], visualize_weight_list[0][0]
                    
                    new_beams.append((new_caption, next_score))

            # 保留分數最高的 beam_width 個候選
            visualize_weight_list = [(visualize_weight_list[i][0],new_beams[i][1]) for i in range(5)]
            new_beams.sort(key=lambda x: x[1], reverse=True)
            visualize_weight_list.sort(key = lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # 選擇最佳的候選
        return beams[0][0][0, len(encoded_caption):], visualize_weight_list[0][0]

class Decoder_with_prefix(nn.Module):

    def __init__(self, cfg, n_prefix=10):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.n_prefix = n_prefix
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block_with_cross_attention(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)
        self.prefix_tokens = nn.Parameter(torch.randn(n_prefix, cfg.n_embd))

        # freeze the weights of the decoder but prefix tokens
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_tokens.requires_grad = True
        for block in self.transformer.h:
            for param in block.cross_attn.parameters():
                param.requires_grad = True

    def forward(self, x: Tensor, feature: Tensor):
        prefix = self.prefix_tokens.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size - self.n_prefix))
        pos = torch.arange(prefix.size()[1] + x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = torch.cat((prefix, self.transformer.wte(x)), dim=1) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x, feature)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x
    
    def caption_image(self, encoded_caption, feature, max_len=57):
        prefix = self.prefix_tokens.unsqueeze(0).repeat(encoded_caption.size(0), 1, 1)
        x = torch.cat((prefix, self.transformer.wte(encoded_caption)), dim=1)
        result_caption = encoded_caption

        for i in range(max_len):
            pos = torch.arange(0, x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
            x_pos = self.transformer.wpe(pos)
            x = x + x_pos
            for block in self.transformer.h:
                x = block(x, feature)
            x = self.transformer.ln_f(x)
            x = self.lm_head(x)
            x = x[:, -1:, :]
            o = torch.argmax(x, dim=-1)

            if o == 50256: # <eos>
                break

            result_caption = torch.cat((result_caption, o), dim=1)
            x = self.transformer.wte(result_caption)

        return result_caption[0, len(encoded_caption)+prefix.size()[1]:]  # Return only the generated part
    
    def caption_image_beam_search(self, encoded_caption, feature, max_len=57, beam_width=5):
        # 初始化 Beam Search
        prefix = self.prefix_tokens.unsqueeze(0).repeat(encoded_caption.size(0), 1, 1)
        beams = [(encoded_caption, 0)]  # (caption, score)

        
        for step in range(max_len):
            new_beams = []
            for caption, score in beams:
                # 相同的 Transformer 操作
                x = torch.cat((prefix, self.transformer.wte(caption)), dim=1)
                pos = torch.arange(0, x.size()[1], device=x.device).unsqueeze(0)
                x = x + self.transformer.wpe(pos)
                for block in self.transformer.h:
                    x = block(x, feature)
                x = self.transformer.ln_f(x)
                x = self.lm_head(x)
                x = x[:, -1, :]

                # 選擇 top-k 候選
                softmax_scores = torch.softmax(x, dim=-1)
                topk_scores, topk_indices = torch.topk(softmax_scores, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # 添加一個額外的維度
                    next_score = score + topk_scores[0, i].log()  # 累計分數
                    new_caption = torch.cat((caption, next_token), dim=1)

                    if next_token == 50256:  # <eos>
                        return new_caption[0, len(encoded_caption):]
                    
                    new_beams.append((new_caption, next_score))

            # 保留分數最高的 beam_width 個候選
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # 選擇最佳的候選
        return beams[0][0][0, len(encoded_caption):]

class Decoder_with_lora(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block_with_lora(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            print(f"loading decoder's checkpoint {self.cfg.checkpoint}")
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, feature: Tensor):
        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        x = self.transformer.wte(x)
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
        x = x + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x, feature)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x

    def caption_image(self, encoded_caption, feature, max_len=57):
        x = self.transformer.wte(encoded_caption)
        result_caption = encoded_caption

        for i in range(max_len):
            pos = torch.arange(0, x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)
            x_pos = self.transformer.wpe(pos)
            x = x + x_pos
            for block in self.transformer.h:
                x = block(x, feature)
            x = self.transformer.ln_f(x)
            x = self.lm_head(x)
            x = x[:, -1:, :]
            o = torch.argmax(x, dim=-1)

            if o == 50256: # <eos>
                break

            result_caption = torch.cat((result_caption, o), dim=1)
            x = self.transformer.wte(result_caption)

        return result_caption[0, len(encoded_caption):]  # Return only the generated part
    
    def caption_image_beam_search(self, encoded_caption, feature, max_len=57, beam_width=5):
        # 初始化 Beam Search
        beams = [(encoded_caption, 0)]  # (caption, score)
        
        for step in range(max_len):
            new_beams = []
            for caption, score in beams:
                # 相同的 Transformer 操作
                x = self.transformer.wte(caption)
                pos = torch.arange(0, x.size()[1], device=x.device).unsqueeze(0)
                x = x + self.transformer.wpe(pos)
                for block in self.transformer.h:
                    x = block(x, feature)
                x = self.transformer.ln_f(x)
                x = self.lm_head(x)
                x = x[:, -1, :]

                # 選擇 top-k 候選
                softmax_scores = torch.softmax(x, dim=-1)
                topk_scores, topk_indices = torch.topk(softmax_scores, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)  # 添加一個額外的維度
                    next_score = score + topk_scores[0, i].log()  # 累計分數
                    new_caption = torch.cat((caption, next_token), dim=1)

                    if next_token == 50256:  # <eos>
                        return new_caption[0, len(encoded_caption):]
                    
                    new_beams.append((new_caption, next_score))

            # 保留分數最高的 beam_width 個候選
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # 選擇最佳的候選
        return beams[0][0][0, len(encoded_caption):]