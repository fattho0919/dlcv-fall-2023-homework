import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *
import sys

torch.backends.cudnn.benchmark = True

metadata_path = sys.argv[1]
output_image_path = sys.argv[2]

ckpt_path = "./nerf_pl/hw4_checkpoints/psnr_3878_ssim_0988_embedding_10_6_fp_128_ep_2.ckpt"

@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    chunk = 1024*32
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back=white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    w, h = [256, 256]

    kwargs = {'root_dir': metadata_path,
              'split': "test"}
    dataset = dataset_dict["hw4_dataset"](**kwargs)

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 6)
    nerf_coarse = NeRF(in_channels_xyz=3+3*10*2, in_channels_dir=3+3*6*2)
    nerf_fine = NeRF(in_channels_xyz=3+3*10*2, in_channels_dir=3+3*6*2)
    load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')
    nerf_coarse.cuda().eval()
    nerf_fine.cuda().eval()

    models = [nerf_coarse, nerf_fine]
    embeddings = [embedding_xyz, embedding_dir]

    imgs = []
    psnrs = []

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        image_name = sample['image_name']
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    64, 128, False,
                                    32*1024*4,
                                    False)

        img_pred = results['rgb_fine'].view(h, w, 3).cpu().numpy()
        

        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(output_image_path, image_name), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')