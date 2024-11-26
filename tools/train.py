# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys

from mmengine._strategy import FSDPStrategy, DDPStrategy
from mmengine._strategy.fsdp import FSDP
from mmengine.config import Config, DictAction
from mmengine.dist import get_rank
from mmengine.runner import FlexibleRunner
sys.path.append(os.curdir)
from mmotion.utils.logger import print_colored_log
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

def load_optim_state_dict(self, state_dict: dict) -> None:
    """Load optimizer state from dict. The original loading function has bug with
    FSDP in torch 2.1.2

    Args:
        state_dict (dict): The optimizer state dict. If ``state_dict_type``
            is ``full``. ``state_dict`` could be the result of
            ``optimizer.state_dict()``
    """
    optim_state_dict = FSDP.optim_state_dict_to_load(
        model=self.model,
        optim_state_dict=state_dict,
        optim=self.optim_wrapper.optimizer)
    self.optim_wrapper.load_state_dict(optim_state_dict)


def load_checkpoint(self, filename: str, **kwargs) -> dict:
    # avoid loading checkpoint to gpu, it causes OOM!
    kwargs['map_location'] = 'cpu'

    if self._is_full_state_dict():
        return super(DDPStrategy, self).load_checkpoint(filename, **kwargs)
    else:
        rank = get_rank()
        filename = osp.join(filename, f'rank{rank}.pth')
        return super(DDPStrategy, self).load_checkpoint(filename, **kwargs)

FSDPStrategy.load_optim_state_dict = load_optim_state_dict
FSDPStrategy.load_checkpoint = load_checkpoint

# FSDPStrategy
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume', action='store_true', help='Whether to resume checkpoint.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir:  # none or empty str
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        if ('constructor' not in cfg.optim_wrapper) or \
                cfg.optim_wrapper['constructor'] == 'DefaultOptimWrapperConstructor':  # noqa
            optim_wrapper = cfg.optim_wrapper.type
            if optim_wrapper == 'AmpOptimWrapper':
                print_colored_log(
                    'AMP training is already enabled in your config.',
                    logger='current',
                    level=logging.WARNING)
            else:
                assert optim_wrapper == 'OptimWrapper', (
                    '`--amp` is only supported when the optimizer wrapper '
                    f'`type is OptimWrapper` but got {optim_wrapper}.')
                cfg.optim_wrapper.type = 'AmpOptimWrapper'
                cfg.optim_wrapper.loss_scale = 'dynamic'
        else:
            for key, val in cfg.optim_wrapper.items():
                if isinstance(val, dict) and 'type' in val:
                    assert val.type == 'OptimWrapper', (
                        '`--amp` is only supported when the optimizer wrapper '
                        f'`type is OptimWrapper` but got {val.type}.')
                    cfg.optim_wrapper[key].type = 'AmpOptimWrapper'
                    cfg.optim_wrapper[key].loss_scale = 'dynamic'

    if args.resume:
        cfg.resume = True

    # build the runner from config
    runner = FlexibleRunner.from_cfg(cfg)

    print_colored_log(f'Working directory: {cfg.work_dir}')
    print_colored_log(f'Log directory: {runner.log_dir}')

    # start training
    runner.train()

    print_colored_log(f'Log saved under {runner.log_dir}')
    print_colored_log(f'Checkpoint saved under {cfg.work_dir}')

if __name__ == '__main__':
    main()
