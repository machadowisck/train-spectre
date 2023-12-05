import os, sys
import numpy as np
import yaml
import torch.backends.cudnn as cudnn
import torch
import shutil
import random
from src.trainer_spectre import Trainer
from src.spectre import SPECTRE
from config import parse_args

def fixed_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(cfg):
    # creat folders
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)

    if cfg.test_mode is False:
        os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
        with open(os.path.join(cfg.output_dir, 'full_config.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # cudnn related setting
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # start training

    spectre = SPECTRE(cfg)

    trainer = Trainer(model=spectre, config=cfg)

    if cfg.test_mode:
        trainer.prepare_data()
        trainer.evaluate(trainer.test_datasets)
    else:
        trainer.fit()

if __name__ == '__main__':
    fixed_seed()
    cfg = parse_args()
    cfg.exp_name = cfg.output_dir

    main(cfg)
