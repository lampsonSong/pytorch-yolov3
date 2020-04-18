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
from utils.utils import non_max_suppression, scale_coords, clip_coords

from coco_eval import convert_out_format, save_json, get_coco_eval, coco80_to_coco91_class

hyp = {
    'weight_decay' : 0.00484,
    'reg_loss_gain' : 0.54,
    'obj_loss_gain' : 64.3,
    'cls_loss_gain' : 37.4,
    'train_iou_thresh' : 0.225,
    'fl_gamma' : 2.
}

def get_dataloader(args, local_rank):
    if not args.test_only:
        # train dataloader
        train_dataset = COCODatasetYolo(
                coco_dir=args.coco_dir, 
                set_name='train2017', 
                img_size=args.img_size,
                multiscale=args.multi_scale_training,
                phase='Train'
                )
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
    # number of classes from dataset
    args.num_classes = test_dataset.num_classes
    test_sampler = DistributedSampler(
            test_dataset,
            num_replicas = args.world_size,
            rank = local_rank
            )
    test_loader = DataLoader(
            dataset = test_dataset, 
            batch_size = 1,
            num_workers = args.num_workers,
            shuffle = False,
            pin_memory = True,
            #sampler = test_sampler, # only one device is used to test
            collate_fn = test_dataset.collate_fn
            )

    if args.test_only:
        return test_loader
    else:
        return train_loader, test_loader

def train_yolo(gpu, args):
    local_rank = args.node_rank * args.gpus + gpu

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
    if args.test_only:
        test_loader = get_dataloader(args, local_rank)
    else:
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

    # optimizer use sgd
    optimizer = optim.SGD(param_g0, lr=args.lr, momentum=args.momentum)
    ## optimizer use adam
    #optimizer = optim.Adam(param_g0, lr=args.lr)

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
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location="cuda:{}".format(local_rank))
        if args.world_size > 1:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
            #model.load_state_dict(checkpoint['model'])

        start_epoch = checkpoint['epoch']



    # scheduler, cosine 
    lr_func = lambda x: (1 + math.cos(x * math.pi / args.epoches)) / 2 * 0.99 + 0.01 
    scheduler =  lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    scheduler.last_epoch = start_epoch

    results = (0,0,0,0,0,0,0)
    model.hyp = hyp
    best_mAP = 0.
    # start training
    for epoch in range(start_epoch, args.epoches):
        if not args.test_only:
            model.train()
            model.iou_ratio = 1. - (1 + math.cos(min(epoch*2, args.epoches) * math.pi / args.epoches)) / 2

            mean_loss = torch.zeros(4)
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))

            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            for idx, (imgs, targets, _, _) in pbar:
                if(len(targets) == 0):
                    continue

                ni = idx + epoch * len(train_loader)

                imgs = imgs.cuda(gpu, non_blocking=True).float() / 255.
                targets = targets.cuda(gpu, non_blocking=True)

                # run model
                yolo_outs = model(imgs)

                loss, loss_items = get_yolo_loss(model, yolo_outs, targets, regression_loss_type = 'GIoU')

                #print(" - loss_items : ", loss_items)
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss_items)
                    return results

                loss /= args.accumulate

                if args.mixed_precision:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # optimize accumulated gradients
                if ni % args.accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # mean loss
                mean_loss = (mean_loss * idx + loss_items.cpu()) / (idx + 1) 
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                #s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, args.epoches - 1), mem, *mean_loss, len(targets), max(imgs[0].shape[2:]))
                s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, args.epoches - 1), mem, *loss_items.cpu(), len(targets), max(imgs[0].shape[2:]))

                pbar.set_description(s)

            # update scheduler
            scheduler.step()

        '''
        Finished one epoch, save model
        1. test coco mAP
        2. save the better model
        '''
        # test coco
        test_device = next(model.parameters()).device
        if str(test_device) == "cuda:0":
            test_model = YOLOv3_SPP(num_classes = args.num_classes)
            test_model.to(test_device)
            
            if args.test_only:
                checkpoint = torch.load(args.resume, map_location=test_device)
                test_model.load_state_dict(checkpoint['state_dict'])
            else:
                if args.world_size > 1:
                    test_model.load_state_dict(model.module.state_dict())
                else:
                    test_model.load_state_dict(model.state_dict())
            
            test_model.eval()
            test_input_shape = (416, 416)
            results = []
            processed_ids = []
            coco91cls = coco80_to_coco91_class()
            tbar = tqdm(enumerate(test_loader), total=len(test_loader))
            for idx, (imgs, targets, orig_imgs_tuple, img_id_tuple) in tbar:
                if(len(targets) == 0):
                    continue

                imgs = imgs.cuda(gpu, non_blocking=True).float() / 255.
                targets = targets.cuda(gpu, non_blocking=True)

                # run model
                with torch.no_grad():
                    thres_out = 0.05
                    yolo_outs = test_model(imgs)
                    outputs = non_max_suppression(yolo_outs, conf_thres=thres_out)
                    for i, det in enumerate(outputs):
                        if det == None:
                            print("det is none")
                            continue

                        orig_img = orig_imgs_tuple[i]
                        det[:, :4] = scale_coords(test_input_shape, det[:, :4], orig_img.shape).round()
                        img_result = convert_out_format(img_id_tuple[i], det, coco91cls, thres_out)
                        if img_result:
                            processed_ids.append(img_id_tuple[i])
                            #results.append(item for item in list(img_result))
                            for item in list(img_result):
                                results.append(item)
                        ##print(results)
                        ## uncommet to vis results
                        #import cv2
                        #for x1, y1, x2, y2, conf, cls in det:
                        #    cv2.rectangle( orig_img, (x1, y1), (x2, y2), (0,0,255), 2)

                        #cv2.imshow("- i", orig_img)
                        #cv2.waitKey(0)

            # save results for new test
            pred_file = "../results/val2017_bbox_results.json"
            save_json(pred_file, results)
            val_file = os.path.join(args.coco_dir,'annotations/instances_val2017.json')

            test_status = get_coco_eval(val_file, pred_file, processed_ids)

            if args.test_only:
                continue
            else:
                # save model
                if local_rank == 0 and not test_status[0] > best_mAP:
                    best_mAP = test_status[0]
                    if args.world_size > 1:
                        state_dict = model.module.state_dict()
                    else:
                        state_dict = model.state_dict()

                    state = {
                            'state_dict' : state_dict,
                            #'optimizer' : optimizer.state_dict(),
                            'epoch' : epoch,
                            #'scheduler' : scheduler.state_dict()
                            }
                    torch.save(state, "../weights/yolov3_epoch{}.pth".format(epoch))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_dir', type=str, default='../data/coco')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0, help='ranking of the nodes')
    parser.add_argument('--gpus', type=int, default='2', help='number of gpus per node')
    parser.add_argument('--epoches', type=int, default='100', help='number of epoches to run')
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--momentum', type=float, default='0.99')
    parser.add_argument('--multi_scale_training', type=bool, default=True)
    parser.add_argument('--mixed_precision', type=bool, default=True)
    parser.add_argument('--test_only', type=bool, default=False)
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
