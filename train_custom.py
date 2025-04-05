import os
import yaml
import torch
import logging
import argparse
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from utils.builder import ConfigBuilder
from utils.functions import display_results, to_device
from time import perf_counter

logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', '-c', default=os.path.join('configs', 'default.yaml'), help='Path to configuration file', type=str)
args = parser.parse_args()

with open(args.cfg, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader=yaml.FullLoader)

builder = ConfigBuilder(**cfg_params)

logger.info('Building models ...')
model = builder.get_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

logger.info('Building dataloaders ...')
train_dataloader = builder.get_dataloader(split='train')
test_dataloader = builder.get_dataloader(split='test')

logger.info('Checking checkpoints ...')
start_epoch = 0
checkpoint_file = os.path.join(builder.get_stats_dir(), 'checkpoint.tar')   
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info(f"Checkpoint {checkpoint_file} (epoch {start_epoch}) loaded.")

logger.info('Building optimizer and learning rate schedulers ...')
optimizer = builder.get_optimizer(model, resume=(start_epoch > 0))
lr_scheduler = builder.get_lr_scheduler(optimizer, resume=(start_epoch > 0))

criterion = builder.get_criterion()
metrics = builder.get_metrics()

def train_one_epoch(epoch):
    logger.info(f'Start training epoch {epoch + 1}')
    model.train()
    losses = []
    with tqdm(train_dataloader) as pbar:
        for data_dict in pbar:
            optimizer.zero_grad()
            data_dict = to_device(data_dict, device)
            res = model(data_dict['input'], data_dict['intrinsics'])  # Passing intrinsics to model
            data_dict['pred'] = res
            loss_dict = criterion(data_dict)
            loss = loss_dict['loss']
            #print(torch.unique(data_dict['loss_mask']))
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.8f}")
            losses.append(loss.item())
    logger.info(f'End training epoch {epoch + 1}, mean loss: {np.mean(losses):.8f}')

def test_one_epoch(epoch):
    logger.info(f'Start testing epoch {epoch + 1}')
    model.eval()
    metrics.clear()
    losses = []
    with tqdm(test_dataloader) as pbar:
        for data_dict in pbar:
            data_dict = to_device(data_dict, device)
            with torch.no_grad():
                res = model(data_dict['input'], data_dict['intrinsics'])  # Passing intrinsics to model
                data_dict['pred'] = res
                loss_dict = criterion(data_dict)
                loss = loss_dict['loss']
            pbar.set_description(f"Epoch {epoch + 1}, loss: {loss.item():.8f}")
            losses.append(loss.item())
    logger.info(f'End testing epoch {epoch + 1}, mean loss: {np.mean(losses):.8f}')
    display_results(metrics.get_results(), logger)

def train(start_epoch):
    for epoch in range(start_epoch, builder.get_max_epoch()):
        logger.info(f'--> Epoch {epoch + 1}/{builder.get_max_epoch()}')
        train_one_epoch(epoch)
        test_one_epoch(epoch)
        lr_scheduler.step()
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, os.path.join(builder.get_stats_dir(), f'checkpoint-epoch{epoch}.tar'))

if __name__ == '__main__':
    train(start_epoch)
