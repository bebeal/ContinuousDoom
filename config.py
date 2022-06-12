import argparse
import torch


def get_config():
    config = argparse.ArgumentParser()
    # hyperparameters
    config.add_argument("--num_frames_per_update", type=int, default=4096)
    config.add_argument("--total_frames", type=int, default=4000000)
    config.add_argument("--max_episode_length", type=float, default=512)
    config.add_argument("-lr", default=7e-4)
    config.add_argument("--gamma", type=float, default=0.99)
    config.add_argument("--lmbda", type=float, default=0.95)
    config.add_argument("--clip_param", type=float, default=0.2)
    config.add_argument("--value_clip_param", type=float, default=1000)
    config.add_argument("--mini_batch_size", type=int, default=256)
    config.add_argument("--k_epochs", type=int, default=1),
    config.add_argument("--entropy_coeff", type=float, default=0.00001)
    config.add_argument("--value_coeff", type=float, default=1)
    config.add_argument("--actor_coeff", type=float, default=0.5)
    config.add_argument("--log_std_init", type=float, default=0.0)
    config.add_argument("--down_sample", default=(None, None))
    config.add_argument("--frame_skip", type=int, default=4)
    config.add_argument("--frame_stack", type=int, default=4)
    config.add_argument("--grad_clip", type=float, default=0.5)
    config.add_argument("--hdim", type=int, default=512)
    config.add_argument("--weight_decay", type=float, default=1e-6)

    # configuration
    config.add_argument("--device", type=str, default="cuda")
    config.add_argument("--log_dir", type=str, default="log")
    config.add_argument("--eval_episodes", type=int, default=10)
    config.add_argument("--every", type=int, default=1000)

    config = config.parse_args()
    config.device = torch.device(config.device)

    return config
