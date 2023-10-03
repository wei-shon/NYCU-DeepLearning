from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 64
    image_channels = 3
    train_batch_size = 32
    eval_batch_size = 32
    num_epochs = 150
    start_epoch = 0
    learning_rate = 2e-5
    diffusion_timesteps = 500
    save_image_epochs = 5
    save_model_epochs = 5
    dataset = './iclevr'
    output_dir = f'checkpoints'
    device = "cuda"
    seed = 0
    resume = None


training_config = TrainingConfig()