import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate
from models.experimental import attempt_load
from models.model import Model
from utils.callbacks import Callbacks
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.autobatch import check_train_batch_size
from utils.dataloaders import create_dataloader
from utils.plots import plot_evolve
from utils.autoanchor import check_anchors
from utils.torch_utils import (
    ModelEMA,
    EarlyStopping,
    smart_optimizer,
    de_parallel
)
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_requirements,
    print_args,
    increment_path,
    init_seeds,
    check_img_size,
    one_cycle,
    colorstr,
    labels_to_class_weights,
    labels_to_image_weights,
    strip_optimizer,
    print_mutation
)


def yaml_load(file="data.yaml"):
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def generate_individual(input_ranges, individual_length):
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "dataset/dataset.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "hyper_parameter.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "results", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader, drop last")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    # callbacks.run("on_pretrain_routine_start")

    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    opt.hyp = str(opt.hyp)
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    hyp = hyp.copy()  # for saving hyps to checkpoints

    plots = not evolve and not opt.noplots  # create plots
    cuda = device != "cpu"
    init_seeds(opt.seed + 1, deterministic=True)

    data_dict = yaml_load(opt.data)
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = int(data_dict["nc"])
    names = data_dict["names"]
    print(nc, names)

    # MODEL
    weights = str(weights)
    pretrained = weights.endswith(".pt")
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    if pretrained:
        ckpt = torch.load(weights, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf']
    else:
        def lf(x):
            return (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    ema = ModelEMA(model)

    best_fitness, start_epoch = 0.0, 0

    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    val_loader = create_dataloader(
        val_path,
        imgsz,
        batch_size,
        gs,
        single_cls,
        hyp=hyp,
        cache=None if noval else opt.cache,
        rect=True,
        rank=-1,
        workers=workers * 2,
        pad=0.5,
        prefix=colorstr("val: "),
    )[0]

    check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
    model.half().float()  # pre-reduce anchor precision

    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp["box"] *= 3 / nl  # scale to layers
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler()
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run("on_train_start")
    LOGGER.info(
        f"Image sizes {imgsz} train, {imgsz} val\n"
        f"Using {train_loader.num_workers} dataloader workers\n"
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f"Starting training for {epochs} epochs..."
    )

    for epoch in range(start_epoch, epochs):
        start_time = time.time()  # 记录 epoch 开始的时间
        callbacks.run("on_train_epoch_start")
        model.train()

        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)

        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            # with torch.cuda.amp.autocast('cuda', amp):
            pred = model(imgs)  # forward
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if opt.quad:
                loss *= 4.0

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            pbar.set_description(
                ("%11s" * 2 + "%11.4g" * 5)
                % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
            )
            # end batch ------------------------------------------------------------------------------------------------
            callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
            if callbacks.stop_training:
                return
        epoch_time = time.time() - start_time  # 计算 epoch 所用时间
        print(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f} seconds.")
        # schedular
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers
        scheduler.step()


        callbacks.run("on_train_epoch_end", epoch=epoch)
        ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        if not noval or final_epoch:  # Calculate mAP
            results, maps, _ = validate.run(
                data_dict,
                batch_size=batch_size,
                imgsz=imgsz,
                model=ema.ema,
                single_cls=single_cls,
                dataloader=val_loader,
                save_dir=save_dir,
                plots=False,
                callbacks=callbacks,
                compute_loss=compute_loss,
            )

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        stop = stopper(epoch=epoch, fitness=fi)  # early stop check
        if fi > best_fitness:
            best_fitness = fi
        log_vals = list(mloss) + list(results) + lr

        if (not nosave) or (final_epoch and not evolve):  # if save
            ckpt = {
                "epoch": epoch,
                "best_fitness": best_fitness,
                "model": deepcopy(de_parallel(model)).half(),
                "ema": deepcopy(ema.ema).half(),
                "updates": ema.updates,
                "optimizer": optimizer.state_dict(),
                "opt": vars(opt),
                "date": datetime.now().isoformat(),
            }

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if opt.save_period > 0 and epoch % opt.save_period == 0:
                torch.save(ckpt, w / f"epoch{epoch}.pt")
            del ckpt
        if stop:
            break

    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers
            if f is best:
                LOGGER.info(f"\nValidating {f}...")
                results, _, _ = validate.run(
                    data_dict,
                    batch_size=batch_size,
                    imgsz=imgsz,
                    model=attempt_load(f, device).half(),
                    iou_thres=0.60,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    save_json=False,
                    verbose=True,
                    plots=plots,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )  # val best model with plots

    return results


def main(opt, callbacks=Callbacks()):
    print_args(vars(opt))
    check_requirements(ROOT / "requirements.txt")

    device = opt.device
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mosaic (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),  # segment copy-paste (probability)
        }

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        initial_values = []

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith("hyper_parameter.yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f"Hyperparameter evolution finished {opt.evolve} generations\n"
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f"Usage example: $ python train.py --hyp {evolve_yaml}"
        )


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)