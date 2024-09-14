import argparse
import logging
import math
import traceback

import numpy as np
import torch
import wandb
from rich.logging import RichHandler
from torch import nn
from tqdm import tqdm

from lmc.data.data_stats import SAMPLE_DICT
from lmc.experiment_config import Trainer
from lmc.utils.metrics import AverageMeter, mixup_topk_accuracy, report_results
from lmc.utils.setup_training import (TrainingElements, cleanup,
                                      save_model_opt, setup_experiment)

FORMAT = "%(name)s - %(levelname)s: %(message)s"

logger = logging.getLogger("trainer")
def get_early_iter_ckpt_steps(steps_per_epoch: int, n_ckpts: int = 10):
    """ schedule for checkpoints """
    first_epoch = np.concatenate(([1,2,3,4,5], np.linspace(6, steps_per_epoch, n_ckpts)))
    later_epochs = np.concatenate([np.linspace(ep*steps_per_epoch, (ep+1)*steps_per_epoch, n_ckpts) for ep in range(1, 10, )])
    ckpts = np.concatenate((first_epoch, later_epochs)).astype(int)
    return ckpts

@torch.no_grad()
def test(model, loader, loss_fn, config, iterator: tqdm, device):
    model.eval()
    total_acc, total_topk, cross_entropy = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )

    iterator.reset()
    for ims, targs in loader:
        ims = ims.to(device)
        targs = targs.to(device)
        iterator.update()
        preds = model(ims)
        loss = loss_fn(preds, targs)

        cross_entropy.update(loss.item(), ims.shape[0])
        acc, topk = mixup_topk_accuracy(preds, targs, k=3, avg=True)
        total_acc.update(acc.item(), ims.shape[0])
        total_topk.update(topk.item(), ims.shape[0])

    res = {
        "cross_entropy": cross_entropy.get_avg(percentage=False),
        "accuracy": total_acc.get_avg(percentage=False),
        "top_3_accuracy": total_topk.get_avg(percentage=False),
    }
    iterator.set_postfix(res)
    iterator.refresh()
    return res

def train(config: Trainer):
    training_elements: TrainingElements
    device: torch.device 
    training_elements, device = setup_experiment(config)
    steps_per_epoch = math.ceil(SAMPLE_DICT[config.data.dataset] / config.data.batch_size)
    max_epochs = training_elements.max_steps.get_epoch(steps_per_epoch)
    loss_fn, test_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.trainer.label_smoothing), nn.CrossEntropyLoss()
    global_step: int = 0
    early_iter_ckpt_steps = get_early_iter_ckpt_steps(steps_per_epoch, n_ckpts=10)
    for ep in range(1, max_epochs+1):
        ### train epoch
        log_dct = dict(epoch=ep)
        training_elements.on_epoch_start()
        train_loaders = [iter(el.train_loader) for el in training_elements]
        for ind, batches in enumerate(zip(*train_loaders)):
            if global_step >= training_elements.max_steps.get_step(steps_per_epoch):
                break
            global_step += 1
            for i, (x, y) in enumerate(batches):
                element = training_elements[i]
                i = i + 1
                if element.curr_step >= element.max_steps.get_step(steps_per_epoch):
                    break
                if element.scheduler is None:
                    lr = element.opt.param_groups[0]["lr"]
                else:
                    lr = element.scheduler.get_last_lr()[-1]
                if config.logger.use_wandb:
                    wandb.log({f"lr/model{i}": lr})

                x, y = batches[i-1]
                element.opt.zero_grad()
                x = x.to(device)
                y = y.to(device)
                out = element.model(x)
                loss = loss_fn(out, y)
                targs_perm = None
                loss.backward()

                element.opt.step()
                if element.scheduler is not None:
                    element.scheduler.step()

                # update metrics
                acc, topk = mixup_topk_accuracy(out.detach(), y.detach(), targs_perm, k=3, avg=True)
                element.metrics.update(acc.item(), topk.item(), None, loss.item(), x.shape[0])
                element.curr_step += 1
                save: bool = element.save_freq_step and element.save_freq_step.modulo(element.curr_step, steps_per_epoch) == 0
                save = save or (config.trainer.save_early_iters and element.curr_step in early_iter_ckpt_steps)
                if save:
                    ckpt_name = f"checkpoints/ep-{ep}-st-{element.curr_step}.ckpt"
                    save_model_opt(element.model, element.opt, element.model_dir.joinpath(ckpt_name), step=element.curr_step, epoch=ep, scheduler=element.scheduler)
        
        for i, element in enumerate(training_elements, start=1):
            # save the end of epoch results
            ckpt_name = f"checkpoints/ep-{ep}.ckpt"
            save_model_opt(element.model, element.opt, element.model_dir.joinpath(ckpt_name), step=element.curr_step, epoch=ep, scheduler=element.scheduler)
            element.model.eval()
            if element.curr_step > element.max_steps.get_step(steps_per_epoch):
                continue
            # logging
            log_dct.update( {
                f"model{i}/train/cross_entropy": element.metrics.cross_entropy.get_avg(percentage=False),
                f"model{i}/train/accuracy": element.metrics.total_acc.get_avg(percentage=False),
                f"model{i}/train/top_3_accuracy": element.metrics.total_topk.get_avg(percentage=False),
            })
            # post epoch processing
            # element.train_iterator.set_postfix(double_res)
            # element.train_iterator.refresh()

            ### test
            with torch.no_grad():
                test_res = test(element.model, element.test_loader, test_loss_fn, config, element.test_iterator, device)
                log_dct.update( {
                            f"model{i}/test/accuracy": test_res["accuracy"],
                            f"model{i}/test/top_3_accuracy": test_res["top_3_accuracy"],
                            f"model{i}/test/cross_entropy": test_res["cross_entropy"],
                        })

                if (test_acc := test_res["accuracy"]) > element.optimal_acc:
                    element.optimal_acc = test_acc
                    if config.trainer.save_best:
                        logger.info(f"Saving best params at epoch {ep}")
                        if element.optimal_path is not None:
                            element.optimal_path.unlink()
                        element.optimal_path = element.model_dir.joinpath(
                            "checkpoints", f"optimal_params-epoch-{ep}.ckpt"
                        )
                        save_model_opt(element.model, element.opt, element.optimal_path, epoch=ep, scheduler=element.scheduler )
        if config.logger.use_wandb:
            wandb.log(log_dct)
        if config.logger.print_summary and log_dct:
            report_results(log_dct, ep, config.n_models)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("subcommand")
    parser.add_argument("--level",choices=["debug", "info", "warning", "error", "critical"], default= "info", help="")
    Trainer.add_args(parser)

    args = parser.parse_args()
    logging.basicConfig(level=args.level.upper(), format=FORMAT, handlers=[RichHandler(show_time=False)])
    platform = Trainer.create_from_args(args)
    print(platform.display)
    try:
        train(platform)
    except Exception as e:
        traceback.print_exc()
        pass
    cleanup(platform)