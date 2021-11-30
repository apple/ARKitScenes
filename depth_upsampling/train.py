import argparse
import os
import shlex
import subprocess
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

import transfroms
from dataset import ARKitScenesDataset
from logs.eval import eval_log
from logs.train import train_log
from losses import get_loss
from models import get_network
from sampler import MultiEpochSampler
from data_utils import batch_to_cuda

TENSORBOARD_DIR = 'tensorboard'


def main(args):
    batch_size = args.batch_size
    num_iter = args.num_iter
    upsample_factor = args.upsample_factor
    start_itr = 0

    patch_size = 256 if args.upsample_factor == 2 else 512

    print('loading train dataset')
    transform = Compose([transfroms.RandomCrop(height=patch_size, width=patch_size, upsample_factor=upsample_factor),
                         transfroms.RandomFilpLR(),
                         transfroms.ValidDepthMask(gt_low_limit=0.01),
                         transfroms.AsContiguousArray()])
    train_dataset = ARKitScenesDataset(root=args.data_path, split='train',
                                       upsample_factor=upsample_factor, transform=transform)
    sampler = MultiEpochSampler(train_dataset, num_iter, start_itr, batch_size)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size,
                                  sampler=sampler,
                                  num_workers=8 * int(torch.cuda.is_available()),
                                  pin_memory=torch.cuda.is_available(),
                                  drop_last=True)

    print('loading validation dataset')
    transform = Compose([transfroms.ModCrop(modulo=32),
                         transfroms.ValidDepthMask(gt_low_limit=0.01)])
    val_dataset = ARKitScenesDataset(root=args.data_path, split='val',
                                     upsample_factor=upsample_factor, transform=transform)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=8 * int(torch.cuda.is_available()),
                                pin_memory=torch.cuda.is_available())

    print('building the network')
    model = get_network(args.network, upsample_factor)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    cudnn.benchmark = True

    # init logs
    if args.tbp is not None:
        print('starting tensorboard')
        tensorboard_path = os.path.join(args.log_dir, TENSORBOARD_DIR)
        command = f'tensorboard --logdir {tensorboard_path} --port {args.tbp}'
        tensorboard_process = subprocess.Popen(shlex.split(command), env=os.environ.copy())
        train_tensorboard_writer = SummaryWriter(os.path.join(tensorboard_path, 'train'), flush_secs=30)
        val_tensorboard_writer = SummaryWriter(os.path.join(tensorboard_path, 'val'), flush_secs=30)
    else:
        print('no tensorboard')
        tensorboard_process = None
        train_tensorboard_writer = None
        val_tensorboard_writer = None

    loss_fn = get_loss(args.network)

    start_time = time.time()
    step = 1
    duration = 0
    current_lr = -1
    print("start training")
    for input_batch in train_dataloader:
        before_op_time = time.time()
        input_batch = batch_to_cuda(input_batch)

        optimizer.zero_grad()
        output_batch = model(input_batch)
        loss = loss_fn(output_batch, input_batch)

        if np.isnan(loss.cpu().item()):
            exit('NaN in loss occurred. Aborting training.')

        loss.backward()
        optimizer.step()

        duration += time.time() - before_op_time

        train_log(step=step, loss=loss, input_batch=input_batch, output_batch=output_batch,
                  tensorboard_writer=train_tensorboard_writer, current_lr=current_lr)
        if step % args.eval_freq == 0:
            eval_log(step, model, val_dataloader, val_tensorboard_writer)

        if step and step % args.log_freq == 0:
            examples_per_sec = args.batch_size / duration * args.log_freq
            time_sofar = (time.time() - start_time) / 3600
            training_time_left = (num_iter / step - 1.0) * time_sofar
            print_string = 'examples/s: {:4.2f} | time elapsed: {:.2f}h | time left: {:.2f}h'
            print(print_string.format(examples_per_sec, time_sofar, training_time_left))
            duration = 0

        if step % args.save_freq == 0:
            checkpoint = {'step': step,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            save_file = os.path.join(args.log_dir, 'checkpoint_step-{}'.format(step))
            torch.save(checkpoint, save_file)

        step += 1

    print('finished training')
    if tensorboard_process is not None:
        tensorboard_process.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth upsamling training', fromfile_prefix_chars='@')

    # data
    parser.add_argument('--data_path', type=str, help='The path to the dataset', default='~/ARKitScenes')

    # Network
    parser.add_argument('--network', type=str, help='network model class', required=True)

    # Losses
    parser.add_argument('--loss', type=str, help='loss for training', action='append')

    # Log and save
    parser.add_argument('--log_dir', type=str, help='directory to save checkpoints and summaries', default='log')
    parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default=None)
    parser.add_argument('--log_freq', type=int, help='log frequency in steps', default=1000)
    parser.add_argument('--eval_freq', type=int, help='run evaluation frequency in steps', default=10000)
    parser.add_argument('--save_freq', type=int, help='Checkpoint saving frequency in steps', default=20000)
    parser.add_argument('--tbp', type=int, help='tensorboard port', default=None)

    # Training
    parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=5e-5)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--num_iter', type=int, help='number of iteration to train', default=200000)
    parser.add_argument('--upsample_factor', type=int, help='upsample scale from low to high resolution',
                        choices=[2, 4, 8])

    args = parser.parse_args()
    main(args)
