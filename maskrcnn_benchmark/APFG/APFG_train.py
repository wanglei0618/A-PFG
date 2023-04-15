import torch

import time
import datetime

from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.logger import debug_print
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.solver.lr_scheduler import WarmupMultiStepLR
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.APFG.Feats_Dataset import Feats_DataSet
from maskrcnn_benchmark.APFG.APFG_model import APFG_MODEL, loss_function
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def APFG_train(cfg, local_rank, distributed, logger):
    APFG_model = APFG_MODEL(cfg)

    debug_print(logger, 'end model construction')
    device = torch.device(cfg.MODEL.DEVICE)
    APFG_model.to(device)
    optimizer = torch.optim.Adam(APFG_model.parameters(), lr=cfg.SOLVER.BASE_LR_APFG, betas=(0.9, 0.999))
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.APFG_STEPS, 0.1, warmup_factor=0.01, warmup_iters=10)
    debug_print(logger, 'end optimizer and shcedule')

    if distributed:
        APFG_model = torch.nn.parallel.DistributedDataParallel(
            APFG_model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')

    arguments = {}
    arguments["epoch"] = 0
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(cfg, APFG_model,optimizer,scheduler, save_dir=cfg.OUTPUT_DIR_APFG,
                                         save_to_disk=save_to_disk, custom_scheduler=True)

    if not cfg.MODEL.APFG_train:
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                  update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)

        return APFG_model

    w_kl = cfg.SOLVER.weight_kl
    w_da = cfg.SOLVER.weight_da
    w_cr = cfg.SOLVER.weight_cr
    arguments["w_kl"] = w_kl
    arguments["w_da"] = w_da
    arguments["w_cr"] = w_cr

    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                  update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
        scheduler.last_epoch = arguments["epoch"]
        w_kl = arguments["w_kl"]
        w_da = arguments["w_da"]
        w_cr = arguments["w_cr"]

    data_set = Feats_DataSet(cfg, logger)
    num_gpus = get_world_size()
    num_batch = cfg.SOLVER.APFG_batch // num_gpus
    train_data_loader = make_APFG_data_loader(cfg, dataset=data_set, batch_size=num_batch,
                                                    shuffle=True, is_distributed=distributed)

    logger.info("Start training")

    meters = MetricLogger(delimiter="  ")
    start_epoch = arguments["epoch"]

    max_epoch = cfg.SOLVER.APFG_epoch
    start_training_time = time.time()
    end = time.time()
    print_first_grad = True

    for epoch in range(start_epoch, max_epoch):
        data_time = time.time() - end
        arguments["epoch"] = epoch
        for iteration, (feats, fg_union_feats, rels) in enumerate(train_data_loader):
            APFG_model.train()
            fg_union_feats = fg_union_feats.to(device)
            feats = feats.to(device)
            rels = rels.type(torch.int64)
            rels = rels.to(device)

            o1_one_hot = torch.zeros(rels.shape[0], 151).to(device)
            o2_one_hot = torch.zeros(rels.shape[0], 151).to(device)
            rels_one_hot0 = torch.zeros(rels.shape[0], 51).to(device)
            o1_one_hot[list(range(rels.shape[0])), rels[:, 1]] = 1
            o2_one_hot[list(range(rels.shape[0])), rels[:, 2]] = 1
            rels_one_hot0[list(range(rels.shape[0])), rels[:, 0]] = 1
            rels_one_hot = torch.cat([rels_one_hot0, o1_one_hot, o2_one_hot], dim=1)

            results = APFG_model(feats, rels_one_hot, fg_union_feats)
            loss = loss_function(results, w_kl, w_da, w_cr)
            loss_all = loss['loss']
            loss_recons = loss['Recon_Loss']
            loss_kld = loss['KLD']
            loss_CA = loss['CA_Loss']
            loss_DA = loss['DA_Loss']
            meters.update(loss=loss_all, loss_recons=loss_recons, loss_CA=loss_CA, loss_DA=loss_DA, loss_kld=loss_kld)
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            verbose = print_first_grad  # print grad or not
            print_first_grad = False
            clip_grad_norm([(n, p) for n, p in APFG_model.named_parameters() if p.requires_grad],
                           max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        w_kl += cfg.SOLVER.weight_kl_delata
        w_da += cfg.SOLVER.weight_da_delata
        w_cr += cfg.SOLVER.weight_cr_delata
        arguments["w_kl"] = w_kl
        arguments["w_da"] = w_da
        arguments["w_cr"] = w_cr
        epoch_time = time.time() - end
        end = time.time()
        meters.update(time=epoch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_epoch- epoch)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        logger.info(
            meters.delimiter.join(
                [
                    "eta: {eta}",
                    "epoch: {epoch}",
                    "{meters}",
                    "lr: {lr:.6f}",
                    "max mem: {memory:.0f}",
                ]
            ).format(
                eta=eta_string,
                epoch=epoch,
                meters=str(meters),
                lr=optimizer.param_groups[-1]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            )
        )

        scheduler.step()
        if (epoch+1) % 20 == 0:
            print_first_grad = True
            checkpointer.save("model_{:07d}".format(epoch), **arguments)

        if epoch == max_epoch:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )

    with open(cfg.OUTPUT_DIR_APFG + "/End_traning", 'w') as f:
        f.write('End training')

    return APFG_model


def make_APFG_data_loader(cfg, dataset=None, batch_size=None, shuffle=True, is_distributed=False):
    images_per_gpu = batch_size
    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(dataset, sampler, False, images_per_gpu)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
    )
    return data_loader