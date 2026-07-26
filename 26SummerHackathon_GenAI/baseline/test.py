import argparse
import json
import os
import subprocess
import sys
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode


def save_one_csv(correct, predn, file, name):
    # 确保文件后缀为 .csv
    base, ext = os.path.splitext(file)
    if ext.lower() != '.csv':
        file = base + '.csv'
    
    # 统计同一个name中的所有cls及其数量
    cls_counter = Counter()
    for i in range(predn.size(0)):
        *_, cls = predn[i, :].tolist()
        if any(correct[i, :]):
            cls_counter[int(cls)] += 1
    
    # 获取所有可能的cls列（可以从现有数据或全局配置中获取）
    # 这里假设我们需要保存所有可能的cls，从当前数据中提取
    all_cls = sorted(cls_counter.keys())
    
    # 准备数据行，第一列为name，后续每列为一个cls及其数量
    data_row = [name]
    clses = []
    for _cls in range(9):
        clses.append(str(cls_counter.get(_cls, '0')))
    data_row.append(';'.join(clses))
    # for cls in all_cls:
    #     data_row.append(cls_counter.get(cls, 0))
    
    # 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.isfile(file)
    try:
        # 写入 CSV 文件
        with open(file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 如果文件是新建的，写入表头
            if not file_exists:
                headers = ['pic_name', 'results']
                writer.writerow(headers)
            
            writer.writerow(data_row)
        
    except FileNotFoundError:
        print(f"错误：找不到路径或文件 - {file}")
    except PermissionError:
        print(f"错误：没有权限写入文件 - {file}")
    except Exception as e:
        print(f"发生未知错误: {e}")  


def process_batch(detections, labels, iouv):
    # labels     = [class_id, x1, y1, x2, y2]
    # detections = [x1, y1, x2, y2, confidence, class_id]
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]))  # IoU > threshold (and classes match)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    metric_only=False,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU thresholdw
    max_det=300,  # maximum detections per image
    task="test",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_csv=False,  # save results to *.txt
    project=ROOT / "results/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
):

    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_csv else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    half = model.fp16  # FP16 supported on limited backends with CUDA
    if engine:
        batch_size = model.batch_size
    else:
        device = model.device
        if not (pt or jit):
            batch_size = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

    # Data
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if pt and not single_cls:  # check --weights are trained on --data
        ncm = model.model.nc
        assert ncm == nc, (
            f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
            f"classes). Pass correct combination of --weights and --data that are trained together."
        )
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
    task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(device=device), Profile(device=device), Profile(device=device)  # profiling times
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    _dict = {}
    csv_results = []
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        preds, _ = model(im) if compute_loss else (model(im, augment=augment), None)

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = []  # for autolabelling
        preds = non_max_suppression(
            preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
        )

        # Metrics
        for si, pred in enumerate(preds):
            _cls = pred[:, -1]
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_csv:
                csv_results.append((path.stem + ".jpg", correct, predn))
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    # Write CSV in sorted order
    if save_csv:
        (save_dir / "labels").mkdir(parents=True, exist_ok=True)
        csv_results.sort(key=lambda x: x[0])
        for name, correct, predn in csv_results:
            save_one_csv(correct, predn, file=save_dir / "labels" / "res.csv", name=name)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or nc < 50) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Return results
    model.float()  # for training
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_csv else ""
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "dataset/dataset.yaml", help="dataset.yaml path")
    parser.add_argument("--metric-only", action="store_true", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "dummy.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="test", help="train, val, test, speed or study")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-csv", action="store_true", help="save results to *.txt")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")

    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
