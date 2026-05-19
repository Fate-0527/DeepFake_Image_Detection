import json
import os
import cv2
import time
import shutil
import random
import datetime
import argparse
import numpy as np
import warnings
import logging as logger
import csv

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.nn.utils import clip_grad_norm_

import albumentations as A
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, average_precision_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import importlib
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate

# [중요] Config 불러오기
from config import Config

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

test_best = -1
test_best_close = -1


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------
# 시각화 및 Top-k 저장 유틸리티 함수
# ---------------------------------------------------------
def plot_metrics(history, save_dir):
    epochs = range(1, len(history['acc']) + 1)
    plt.figure(figsize=(15, 10))
    
    metrics = ['loss', 'acc', 'f1', 'auc'] 
    titles = ['Val Loss', 'Accuracy', 'F1-Score', 'AUC']
    colors = ['r', 'b', 'm', 'g'] 
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        if metric in history and len(history[metric]) > 0:
            plt.subplot(2, 2, i + 1)
            plt.plot(epochs, history[metric], f'{color}o-', label=title)
            plt.title(f'{title} Trend')
            plt.xlabel('Epochs')
            plt.ylabel('Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            if len(epochs) < 20: plt.xticks(epochs)
    
    plt.tight_layout()
    
    # Config에 정의된 파일명으로 저장
    save_path = os.path.join(save_dir, Config.PNG_FILE)
    plt.savefig(save_path)
    plt.close()

def sort_top_k(top_k_list, candidate, k=3):
    # candidate: (Val_Loss, Filename) 튜플
    if len(top_k_list) == 0:
        return [candidate], []
    
    new_list = top_k_list + [candidate]
    # Loss 기준 오름차순 정렬 (낮은 게 좋음)
    new_list.sort(key=lambda x: x[0], reverse=False) 
    
    if len(new_list) > k:
        to_be_del = new_list[k:] # 순위 밖 삭제
        top_k_list = new_list[:k]
    else:
        to_be_del = []
        top_k_list = new_list
    return top_k_list, to_be_del
# ---------------------------------------------------------


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)
        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file,
                 data_size=512, val_ratio=None, split_anchor=True,
                 args=None,
                 map_file=None):
        self.data_root = data_root
        self.data_size = data_size
        self.train_list = []
        self.anchor_list = []
        self.isAnchor = False
        self.isVal = False
        self.split_anchor = split_anchor
        
        if map_file is None and args is not None:
            self.map_file = args.map_file
        else:
            self.map_file = map_file
            
        if not self.map_file or not os.path.exists(self.map_file):
            print(f"Warning: Map file not found at {self.map_file}. Using dummy maps.")

        self.albu_pre_train = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        self.albu_pre_train_easy = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)
        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.args = args

        with open(train_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            try:
                image_path, image_label = line.split('\t')
            except ValueError:
                image_path, image_label = line.rsplit(' ', 1)
                
            label = int(image_label)
            if self.split_anchor and random.random() < 0.1 and label == 0 and len(self.anchor_list) < 100:
                self.anchor_list.append((image_path, label))
            else:
                self.train_list.append((image_path, label))

        if val_ratio is not None:
            np.random.shuffle(self.train_list)
            self.test_list = self.train_list[-int(len(self.train_list) * val_ratio):]
            self.train_list = self.train_list[:-int(len(self.train_list) * val_ratio)]
        else:
            self.test_list = self.train_list

        filename_to_loss = {}
        if self.map_file and os.path.exists(self.map_file):
            with open(self.map_file) as f:
                for line in f:
                    try:
                        path, _ = line.strip().split('\t')
                    except:
                        parts = line.strip().split(' ')
                        path = parts[0]
                    
                    filename = path.split('/')[-1].split('.')[0]
                    filename_to_loss[filename] = path

        ordered_map_paths = []
        for ann in self.train_list:
            image_path = ann[0]
            filename = image_path.split('/')[-1].split('.')[0]
            if filename in filename_to_loss:
                loss_path = filename_to_loss[filename]
                ordered_map_paths.append(loss_path)
            else:
                ordered_map_paths.append("") 
                
        self.ordered_map_paths = ordered_map_paths

    def transform(self, x):
        if self.isVal:
            x = self.albu_pre_val(image=x)['image']
        else:
            if self.args.no_strong_aug:
                x = self.albu_pre_train_easy(image=x)['image']
            else:
                x = self.albu_pre_train(image=x)['image']
        x = self.imagenet_norm(x)
        return x

    def __len__(self):
        if self.isAnchor:
            return len(self.anchor_list)
        elif self.isVal:
            return len(self.test_list)
        else:
            return len(self.train_list)

    def __getitem__(self, index):
        if self.isAnchor:
            return self.getitem(index, self.anchor_list)
        elif self.isVal:
            return self.getitem(index, self.test_list)
        else:
            return self.getitem(index, self.train_list)

    def getitem(self, index, data_list):
        image_path, onehot_label = data_list[index]
        map_path = self.ordered_map_paths[index]

        if map_path and os.path.exists(map_path):
            try:
                loss_map = torch.load(map_path)
            except:
                loss_map = torch.zeros((4, 32, 32))
        else:
            loss_map = torch.zeros((4, 32, 32)) 

        if not os.path.exists(image_path):
            image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)

        if image is None:
            image = np.zeros([512, 512, 3], dtype=np.uint8)
        image = image[..., ::-1]

        crop = self.transform(image)
        onehot_label = torch.LongTensor([onehot_label])
        return crop, onehot_label, loss_map

    def set_val_True(self):
        self.isVal = True

    def set_val_False(self):
        self.isVal = False

    def set_anchor_True(self):
        self.isAnchor = True

    def set_anchor_False(self):
        self.isAnchor = False


def train_one_epoch(data_loader, model, optimizer, cur_epoch, loss_meter, args, device, writer, ngpus_per_node):
    loss_meter.reset()
    batch_idx = 0
    loss_avg = 0
    for (images, labels, loss_maps) in data_loader:
        images = images.to(device)
        labels = labels.to(device).flatten().squeeze()
        loss_maps = loss_maps.to(device)
        
        logits = model(images, loss_maps)
        loss = args.criterion_ce(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(loss.item(), images.shape[0])
        if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                     and args.rank % ngpus_per_node == 0)) and batch_idx % 50 == 0 and batch_idx > 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info(
                'Ep %03d, it %03d/%03d, lr: %8.7f, CE: %7.6f' % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            loss_meter.reset()
            writer.add_scalar('train/loss', loss_avg, loss_meter.count)
            writer.add_scalar('train/lr', lr, loss_meter.count)
        batch_idx += 1
    logger.info('End Training')
    return loss_avg


def validation_contrastive(model, args, test_file, device, ngpus_per_node):
    logger.info('Start eval')
    model.eval()
    val_dataset = ImageDataset(args.data_root, test_file, data_size=args.data_size, split_anchor=False, args=args, map_file=args.map_file)
    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None
    data_loader = DataLoader(
        val_dataset, args.test_batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    data_loader.dataset.set_val_True()
    data_loader.dataset.set_anchor_False()
    
    gt_labels_list, prob_labels_list = [], []
    val_losses = AverageMeter()
    
    with torch.no_grad():
        for iter, (images, labels, loss_maps) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.flatten().squeeze().to(device)
            loss_maps = loss_maps.to(device)
            
            if len(images.shape) == 5:
                 b, n, c, h, w = images.shape
                 images = images.reshape(b*n, c, h, w)
            
            try:
                logits = model(images, loss_maps)
                loss = args.criterion_ce(logits, labels)
                val_losses.update(loss.item(), images.size(0))
                
                prob = torch.softmax(logits, dim=-1)
            except Exception as e:
                logger.info(f'Bad eval batch: {e}')
                continue
                
            gt_labels_list.append(labels)
            prob_labels_list.append(prob[:, 1])

    if len(gt_labels_list) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    gt_labels = torch.cat(gt_labels_list, dim=0)
    prob_labels = torch.cat(prob_labels_list, dim=0)

    gt_labels_list = gt_labels.cpu().numpy()
    prob_labels_list = prob_labels.cpu().numpy()

    auc = roc_auc_score(gt_labels_list, prob_labels_list)
    ap = average_precision_score(gt_labels_list, prob_labels_list)
    
    best_acc = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        pred = (prob_labels_list > thresh).astype(int)
        acc = accuracy_score(gt_labels_list, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    pred_labels = (prob_labels_list > best_thresh).astype(int)
    
    precision = precision_score(gt_labels_list, pred_labels, zero_division=0)
    recall = recall_score(gt_labels_list, pred_labels, zero_division=0)
    f1 = f1_score(gt_labels_list, pred_labels, zero_division=0)

    real_idx = (gt_labels_list == 0)
    fake_idx = (gt_labels_list == 1)
    
    if np.sum(real_idx) > 0:
        r_acc = accuracy_score(gt_labels_list[real_idx], pred_labels[real_idx])
    else:
        r_acc = 0.0
        
    if np.sum(fake_idx) > 0:
        f_acc = accuracy_score(gt_labels_list[fake_idx], pred_labels[fake_idx])
    else:
        f_acc = 0.0
        
    raw_acc = accuracy_score(gt_labels_list, (prob_labels_list > 0.5).astype(int))
    
    model.train()
    return auc, best_acc, ap, raw_acc, r_acc, f_acc, val_losses.avg, precision, recall, f1


def main(gpu, ngpus_per_node, args):
    global test_best

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(log_dir=os.path.join(args.out_dir))
    else:
        writer = FakeWriter()
        
    args.gpu = gpu
    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = getattr(importlib.import_module('model'), args.model)(num_class=args.num_class, clip_type=args.clip_type)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    device = torch.device(f'cuda:{args.gpu}') if args.gpu is not None else torch.device("cuda")

    train_dataset = ImageDataset(args.data_root, args.train_file, data_size=args.data_size, val_ratio=None,
                                 split_anchor=False, args=args, map_file=args.map_file)
                                 
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        logger.info(f'Train dataset size: {len(train_dataset)}')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.lr)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    loss_meter = AverageMeter()

    history = {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    top_k_checkpoints = [] 
    k_num = 3 

    # CSV 로그 파일 경로 (Config.SAVE_DIR 사용)
    log_file_path = os.path.join(Config.SAVE_DIR, "training_log.csv")
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        with open(log_file_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score'])

    for epoch in range(args.epoches):
        epoch_start = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        train_data_loader.dataset.set_val_False()
        
        if args.isTrain == 1:
            train_loss_avg = train_one_epoch(train_data_loader, model, optimizer, epoch, loss_meter, args, device, writer, ngpus_per_node)
            
            val_auc, val_acc, val_ap, val_raw_acc, val_r_acc, val_f_acc, val_loss, val_pre, val_rec, val_f1 = validation_contrastive(
                model, args, args.val_file, device, ngpus_per_node)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                logger.info(f'Epoch {epoch} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
                writer.add_scalar('val/AUC', val_auc, epoch)
                
                # History 업데이트
                history['loss'].append(val_loss)
                history['acc'].append(val_acc)
                history['precision'].append(val_pre)
                history['recall'].append(val_rec)
                history['f1'].append(val_f1)
                history['auc'].append(val_auc)
                
                # 그래프 그리기 (Config.SAVE_DIR 사용)
                plot_metrics(history, Config.SAVE_DIR)
                
                # CSV 저장
                with open(log_file_path, 'a', newline='') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow([epoch, train_loss_avg, val_loss, val_acc, val_auc, val_pre, val_rec, val_f1])
                
                # Top-k 저장 (Config.FILE_NAME 사용)
                current_filename = f"{Config.FILE_NAME}_ep{epoch}_loss{val_loss:.4f}_acc{val_acc:.4f}.pth"
                candidate = (val_loss, current_filename)
                
                top_k_checkpoints, to_be_del = sort_top_k(top_k_checkpoints, candidate, k=k_num)
                
                if candidate not in to_be_del:
                    save_path = os.path.join(Config.SAVE_DIR, current_filename)
                    torch.save(model.state_dict(), save_path)
                    logger.info(f">>> 💾 Saved Top-{k_num} (Low Loss): {current_filename}")
                    
                    for _, del_filename in to_be_del:
                        del_path = os.path.join(Config.SAVE_DIR, del_filename)
                        if os.path.exists(del_path):
                            os.remove(del_path)
                            logger.info(f">>> 🗑️ Deleted: {del_filename}")

            lr_schedule.step(val_auc)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class FakeWriter:
    def __init__(self): pass
    def add_scalar(self, p1, p2, p3): pass

if __name__ == '__main__':
    conf = argparse.ArgumentParser()
    conf.add_argument("--data_root", type=str, default='')
    conf.add_argument("--train_file", type=str, default='annotation/my_train_list.txt')
    conf.add_argument("--val_file", type=str, default='annotation/my_val_list.txt')
    conf.add_argument("--test_file", type=str, default='')
    conf.add_argument('--val_ratio', type=float, default=0.0)
    conf.add_argument('--isTrain', type=int, default=1)
    conf.add_argument("--model", type=str, default='CLipClassifierWMapV6')
    conf.add_argument("--num_class", type=int, default=2)
    conf.add_argument('--lr', type=float, default=1e-4)
    conf.add_argument('--epoches', type=int, default=100)
    conf.add_argument('--batch_size', type=int, default=16)
    conf.add_argument('--test_batch_size', type=int, default=16)
    conf.add_argument('--data_size', type=int, default=256)
    
    # [수정] Config.SAVE_DIR을 기본값으로 사용
    conf.add_argument("--out_dir", type=str, default=Config.SAVE_DIR)
    
    conf.add_argument("--break_onek", action='store_true', default=False)
    conf.add_argument("--val_method", type=str, default="con")
    conf.add_argument("--no_strong_aug", action='store_true', default=True) 
    conf.add_argument("--label_smooth", action='store_true', default=False)
    conf.add_argument('--smoothing', type=float, default=0.1)
    conf.add_argument("--seed", type=int, default=42)
    conf.add_argument("--gpu", type=int, default=0)
    conf.add_argument("--exp_name", type=str, default='LaRE_Run')
    conf.add_argument("--clip_type", type=str, default='RN50')
    
    conf.add_argument("--map_file", type=str, required=True, help="Path to ann.txt")
    
    conf.add_argument('--multiprocessing-distributed', action='store_true')
    conf.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str)
    conf.add_argument('--dist-backend', default='nccl', type=str)
    conf.add_argument('--world-size', default=1, type=int)
    conf.add_argument('--rank', default=0, type=int)
    conf.add_argument('-j', '--workers', default=4, type=int)
    
    args = conf.parse_args()
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
        
    main(args.gpu, ngpus_per_node, args)