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

def evaluate_train(config, pipeline, model , cond):
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
    test_image = torch.tensor(images).float().permute(0,3,1,2).to(config.device)

    return test_image

def evaluate(config, epoch, pipeline, model , cond):
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

    test_image = torch.tensor(images).float().permute(0,3,1,2).to(config.device)
    test_image = torchvision.transforms.Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))
    test_model = evaluation_model()
    acc = test_model.eval(test_image , cond)

    grid_save_dir = Path(config.output_dir, "samples")
    grid_save_dir.mkdir(parents=True, exist_ok=True)
    image_grid.save(f"{grid_save_dir}/{epoch:04d}.png")

    return acc


if __name__ == '__main__':
    # Dataloader
    train_dataloader = DataLoader(
        iclevrDataset(mode='train', root='iclevr'),
        batch_size=training_config.train_batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        iclevrDataset(mode='test', root='iclevr'),
        batch_size=training_config.eval_batch_size,
        shuffle=False
    )
    
    # training
    # model = UNet(image_size=training_config.image_size,
    #              input_channels=training_config.image_channels).to(training_config.device)

    # model = UNet2DModel(image_size=training_config.image_size,
    #              input_channels=training_config.image_channels).to(training_config.device)
    model = UNet2DModel(
        sample_size=training_config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(training_config.device)

    print("Model size: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                              T_max=len(train_dataloader) * training_config.num_epochs,
                                                              last_epoch=-1,
                                                              eta_min=1e-9)

    if training_config.resume:
        checkpoint = torch.load(training_config.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        training_config.start_epoch = checkpoint['epoch'] + 1

    for param_group in optimizer.param_groups:
        param_group['lr'] = training_config.learning_rate

    diffusion_pipeline = DDPMPipeline(beta_start=1e-6, beta_end=1e-2, num_timesteps=training_config.diffusion_timesteps)

    global_step = training_config.start_epoch * len(train_dataloader)

    # Training loop
    for epoch in range(training_config.start_epoch, training_config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        mean_loss = 0
        loss_fn = nn.MSELoss()
        model.train()
        for step, batch in enumerate(train_dataloader):
            original_images = batch[0].to(training_config.device)
            cond = batch[1].to(training_config.device)
            # print(cond.shape)
            # all = torch.cat([original_images , cond])
            # print(all)
            batch_size = original_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (batch_size,),
                                      device=training_config.device).long()
            
            # Apply forward diffusion process at the given timestep
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noisy_images = noisy_images.to(training_config.device)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps , cond).sample
            # print(noise_pred.shape)
            loss = F.mse_loss(noise_pred, noise)

            # Predict the image loss
            # generate_image = evaluate_train(training_config, diffusion_pipeline, model , cond)
            # loss2 = loss_fn(generate_image , original_images)
            # loss += loss2

            # Calculate new mean on the run without accumulating all the values
            mean_loss = mean_loss + (loss.detach().item() - mean_loss) / (step + 1)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": mean_loss, "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

        # Evaluation
        # if (epoch) % training_config.save_image_epochs == 0 :
        #     model.eval()
        #     for step, batch in enumerate(test_dataloader):
        #         cond = batch.to(training_config.device)
        #         acc = evaluate(training_config, epoch, diffusion_pipeline, model , cond)
        #         print(f"epoch {epoch} test acc : " , acc)

        if (epoch +1) % training_config.save_image_epochs == 0 or epoch == training_config.num_epochs - 1:
            model.eval()
            for step, batch in enumerate(test_dataloader):
                cond = batch.to(training_config.device)
                acc = evaluate(training_config, epoch, diffusion_pipeline, model , cond)
                print(f"epoch {epoch} test acc : " , acc)

        if (epoch + 1) % training_config.save_model_epochs == 0 or epoch == training_config.num_epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'parameters': training_config,
                'epoch': epoch
            }
            torch.save(checkpoint, Path(training_config.output_dir,
                                        f"unet{training_config.image_size}_e{epoch}.pth"))


    # best_acc = 0
    # for epoch in range(args.n_epochs):
    #     Diffusion_model.eval()
    #     with torch.no_grad():
    #         for i, cond in enumerate(test_dataloader):
    #             cond = cond.to(device)
    #             batch_size = cond.size(0)
    #             noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
    #             fake_image = Diffusion_model(noise, cond)
    #             save_image(fake_image.detach(),
    #                 '%s/fake_test_samples_epoch_%03d.png' % (args.outf, epoch),
    #                 normalize=True)
    #             acc = evaluator.eval(fake_image, cond)
    #             # do checkpointing
    #             if acc > best_acc:
    #                 best_acc = acc

