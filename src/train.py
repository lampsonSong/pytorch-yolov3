# created by lampson.song @ 2020-3-23
# training scripts of YOLOv3

import os
import argparse
from dataset.coco_dataset_yolo import COCODatasetYolo

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from net.yolov3_spp import YOLOv3_SPP
from net.yolov3_loss import get_yolo_loss

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import apex

from tqdm import tqdm
import math

hyp = {
        'weight_decay' : 0.0,
        'reg_loss_gain' : 0.5,
        'obj_loss_gain' : 40.0,
        'cls_loss_gain' : 20.0,
        'train_iou_thresh' : 0.225,
        'fl_gamma' : 2.
    }

def get_dataloader(args, local_rank):
    # train dataloader
    train_dataset = COCODatasetYolo(
            coco_dir=args.coco_dir, 
            set_name='val2017', 
            img_size=args.img_size,
            multiscale=args.multi_scale_training,
            phase='Train'
        )
    args.num_classes = train_dataset.num_classes
    train_sampler = DistributedSampler(
            train_dataset,
            num_replicas = args.world_size,
            rank = local_rank
        )
    train_loader = DataLoader(
            dataset = train_dataset, 
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            shuffle = False,
            pin_memory = True,
            sampler = train_sampler,
            collate_fn = train_dataset.collate_fn
        )

    # test dataloader
    test_dataset = COCODatasetYolo(
            coco_dir=args.coco_dir, 
            set_name='val2017', 
            img_size=args.img_size, 
            phase='Test'
        )
    test_sampler = DistributedSampler(
            test_dataset,
            num_replicas = args.world_size,
            rank = local_rank
            )
    test_loader = DataLoader(
            dataset = train_dataset, 
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            shuffle = False,
            pin_memory = True,
            sampler = test_sampler,
            collate_fn = test_dataset.collate_fn
        )

    return train_loader, test_loader



def train_yolo(gpu, args):
    local_rank = args.node_rank * args.gpus + gpu
    print(" - local rank : ", local_rank)

    if torch.cuda.is_available():
        if args.world_size > 1:
            dist.init_process_group(
                    backend = 'nccl', # distributed backend
                    init_method = 'env://',
                    #init_method = 'tcp://127.0.0.1:9998', # distributed training init method
                    world_size = args.world_size, # number of nodes for distributed training
                    rank = local_rank # distributed training node rank
                    )
    else:
        print("CUDA is not available")
        exit(0)

    torch.manual_seed(0)

    # get dataloader
    train_loader, test_loader = get_dataloader(args, local_rank)

    # model initilization
    model = YOLOv3_SPP(num_classes = args.num_classes)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
   
    # add optimizer to parameters of the model
    param_g0, param_g1, param_g2 = [], [], []
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            param_g2 += [v]
        elif 'Conv2d.weight' in k:
            param_g1 += [v] # apply weight decay
        else:
            param_g0 += [v]

    ## optimizer use sgd
    #optimizer = optim.SGD(param_g0, lr=args.lr, momentum=args.momentum)
    ## optimizer use adam
    optimizer = optim.Adam(param_g0, lr=args.lr)
    
    optimizer.add_param_group({'params': param_g1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': param_g2})
    optimizer.param_groups[2]['lr'] *= 2.0 # bias learning rate
    del param_g0, param_g1, param_g2
    
    if args.mixed_precision:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    else:
        if args.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                    model, 
                    device_ids=[gpu],
                    find_unused_parameters=True)

    start_epoch = 0
    # resume TODO
    
    
    # scheduler, cosine 
    lr_func = lambda x: (1 + math.cos(x * math.pi / args.epoches)) / 2 * 0.99 + 0.01 
    scheduler =  lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    scheduler.last_epoch = start_epoch

    results = (0,0,0,0,0,0,0)
    model.hyp = hyp
    # start training
    for epoch in range(start_epoch, args.epoches):
        model.train()
        model.iou_ratio = 1. - (1 + math.cos(min(epoch*2, args.epoches) * math.pi / args.epoches)) / 2

        mean_loss = torch.zeros(4)
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (imgs, targets) in pbar:
            if(len(targets) == 0):
                continue
            
            ni = idx + epoch * len(train_loader)

            imgs = imgs.cuda(gpu, non_blocking=True).float() / 255.
            targets = targets.cuda(gpu, non_blocking=True)

            # run model
            yolo_outs = model(imgs)

            loss, loss_items = get_yolo_loss(model, yolo_outs, targets, regression_loss_type = 'GIoU')
            ### check loss
            #f_obj = torch.nn.BCEWithLogitsLoss(reduction='mean')
            #loss = torch.zeros(1, requires_grad=True).to(yolo_outs[0].device)
            #for y_item in yolo_outs:
            #    l_target = torch.zeros_like(y_item)
            #    loss += f_obj(y_item, l_target)


            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results
            
            loss /= args.accumulate
            #print(" - loss : ", loss)

            if args.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # optimize accumulated gradients
            if ni % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            ## mean loss
            #mean_loss = (mean_loss * idx + loss_items.cpu()) / (idx + 1) 
            #mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            #s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, args.epoches - 1), mem, *mean_loss, len(targets), max(imgs[0].shape[2:]))
            #
            #pbar.set_description(s)

        # update scheduler
        scheduler.step()

        # finished one epoch, save model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_dir', type=str, default='../data/coco')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0, help='ranking of the nodes')
    parser.add_argument('--gpus', type=int, default='2', help='number of gpus per node')
    parser.add_argument('--epoches', type=int, default='100', help='number of epoches to run')
    parser.add_argument('--lr', type=float, default='1e-5')
    parser.add_argument('--momentum', type=float, default='0.99')
    parser.add_argument('--multi_scale_training', type=bool, default=True)
    parser.add_argument('--mixed_precision', type=bool, default=False)
    parser.add_argument('--accumulate', type=int, default=4, help='accumulate 4 times then optimizing')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9997'
        mp.spawn(train_yolo, nprocs=args.gpus, args=(args,))
    else:
        args.mixed_precision = False
        train_yolo(0, args)


    print("done")
