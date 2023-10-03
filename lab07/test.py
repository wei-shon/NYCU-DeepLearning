import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import torch.nn as nn
from dataset import iclevrDataset
from tqdm import tqdm
from evaluator import evaluation_model
from Unet import UNet
from torch.optim import Adam
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

# from src.data.loader import get_loader
from Unet import UNet
from DDPM import DDPMPipeline
from common import postprocess, create_images_grid
from config import training_config
from Unet2D import UNet2DModel
import torchvision

def evaluate(config, pipeline, model , cond , test_name = 'test'):
    # Perform reverse diffusion process with noisy images.
    noisy_sample = torch.randn(
        cond.shape[0],
        config.image_channels,
        config.image_size,
        config.image_size).to(config.device)

    # Reverse diffusion for T timesteps
    images = pipeline.sampling(model, noisy_sample, config.device , cond)

    # Postprocess and save sampled images
    images = postprocess(images)
    image_grid = create_images_grid(images, rows=4, cols=8)

    test_image = torch.tensor(images).permute(0,3,1,2).float().to(config.device)
    test_model = evaluation_model()
    acc = test_model.eval(test_image , cond)

    grid_save_dir = Path(config.output_dir, "samples")
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{test_name}.png")

    return acc


if __name__ == '__main__':
    # Dataloader
    test_new_dataloader = DataLoader(
        iclevrDataset(mode='new_test', root='iclevr'),
        batch_size=training_config.eval_batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        iclevrDataset(mode='test', root='iclevr'),
        batch_size=training_config.eval_batch_size,
        shuffle=False
    )
    
    # training
    model = UNet(image_size=training_config.image_size,
                 input_channels=training_config.image_channels).to(training_config.device)
    
    # model = UNet2DModel(
    #     sample_size=training_config.image_size,  # the target image resolution
    #     in_channels=3,  # the number of input channels, 3 for RGB images
    #     out_channels=3,  # the number of output channels
    #     layers_per_block=2,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    #     down_block_types=(
    #         "DownBlock2D",  # a regular ResNet downsampling block
    #         "DownBlock2D",
    #         "DownBlock2D",
    #         "DownBlock2D",
    #         "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
    #         "DownBlock2D",
    #     ),
    #     up_block_types=(
    #         "UpBlock2D",  # a regular ResNet upsampling block
    #         "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
    #         "UpBlock2D",
    #         "UpBlock2D",
    #         "UpBlock2D",
    #         "UpBlock2D",
    #     ),
    # ).to(training_config.device)

    print("Model size: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                              T_max=len(test_dataloader) * training_config.num_epochs,
                                                              last_epoch=-1,
                                                              eta_min=1e-9)


    resume = "./checkpoints/unet64_e149.pth"
    checkpoint = torch.load(resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    training_config.start_epoch = checkpoint['epoch'] + 1

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = training_config.learning_rate

    diffusion_pipeline = DDPMPipeline(beta_start=1e-4, beta_end=1e-2, num_timesteps=training_config.diffusion_timesteps)
    
    for step, batch in enumerate(test_dataloader):
        cond = batch.to(training_config.device)
        acc = evaluate(training_config, diffusion_pipeline, model , cond , 'test')
        print(f"test acc : " , acc)
    
    for step, batch in enumerate(test_new_dataloader):
        cond = batch.to(training_config.device)
        acc = evaluate(training_config, diffusion_pipeline, model , cond , 'new_test')
        print(f"new_test acc : " , acc)