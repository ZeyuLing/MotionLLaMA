import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from mmengine import Config
from packaging.version import parse

from mmotion.utils.distribute.policies import bfSixteen_mixed, fpSixteen, get_llama_wrapper


def get_policies(cfg:Config, rank):
    """ Get the policies for mixed precision and fsdp wrapping
    :param cfg: fsdp configuration
    :param rank: local_rank
    :return:
    """

    verify_bfloat_support = (torch.version.cuda  \
                            and torch.cuda.is_bf16_supported()\
                            and parse(torch.version.cuda).release >= (11, 0) \
                            and dist.is_nccl_available()and nccl.version() >= (2, 10))

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.get('mixed_precision', False):
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy


def fsdp_auto_wrap_policy(model, transformer_layer_name):
    import functools
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return auto_wrap_policy