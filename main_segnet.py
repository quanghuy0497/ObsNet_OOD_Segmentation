import argparse
import datetime
import os
import pdb
import pprint
import random
import time
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler

import wandb
from configs import argsconfig
from Datasets.load_data import data_loader
from engine_segnet import evaluate, train
from Models.load_net_segnet import net_loader
from Utils.utils import reinitialize
from Utils.metrics import IoU


def main(args):
    """ Manage the training, the testing and the declaration of the 'global parameters'
        such as optimizer, scheduler, the SumarryWriter, etc. """
        
    # Load Dataset
    train_loader, val_loader, test_loader = data_loader(args.data, args)

    args.cmap = train_loader.dataset.cmap
    args.class_name = train_loader.dataset.class_name
    args.colors = np.array(train_loader.dataset.colors)
    
    # Load Network
    segnet = net_loader(args)
    
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(segnet.parameters(), lr=args.lr)
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(segnet.parameters(), lr=args.lr)

    sched = scheduler.MultiStepLR(optimizer, milestones=[args.epoch // 2, args.epoch-5], gamma=0.2)
    
    metric = IoU(args.nclass)

    if args.test:
        # evaluate(0, segnet, test_loader, "Test", metric, args)
        print()
    else:
        best = 0
        start = time.time()
        
        print("===================================")
        print("Training phase: ")
        print("===================================")
        
        for epoch in range(0, args.epoch+1):
            print(f"\n######## Epoch: {epoch} || Time: {(time.time() - start)/60:.2f} min ########")

            train_loss, train_acc = train(epoch, segnet, train_loader, optimizer, args)
            val_loss, val_global_acc, val_class_acc, val_iou = evaluate(epoch, segnet, val_loader, "Val", metric, args)
            
            if args.wandb:
                wandb.log({'Train Loss': train_loss, 'Train Acc': train_acc, 'Val Loss': val_loss, 'Val Acc': val_global_acc, 'Val Class Acc': val_class_acc, 'Val IoU': val_iou}, step = epoch + 1)

            if epoch % 5 == 0:               # save Checkpoint
                model_to_save = segnet.module.state_dict()
                torch.save(model_to_save, os.path.join(args.segnet_file, f"epoch{epoch:03d}.pth"))

            if val_iou > best:  # Save Best model
                print("save best net!!!")
                best = val_iou
                model_to_save = segnet.module.state_dict()
                torch.save(model_to_save, os.path.join(args.segnet_file, "best.pth"))
            sched.step()
        
        # print("===================================")
        # print("Testing phase: ")
        # print("===================================")
        
        # args.test = True
        # segnet = net_loader(args)      
        # metric = IoU(args.nclass)                 
        
        # test_loss, test_global_acc, test_class_acc, test_IoU = evaluate(0, segnet, test_loader, "Test", metric, args)
        # if args.wandb:
        #     wandb.log({'Test Loss': test_loss, 'Test Acc': test_acc})
             
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    ### Argparse ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_folder",     type=str,      default="",         help="path to dataset")
    parser.add_argument("--segnet_file",     type=str,      default="",         help="path to segnet")
    parser.add_argument("--model",           type=str,      default="segnet",   help="segnet|deeplabv3plus|road_anomaly")
    parser.add_argument("--data",            type=str,      default="",         help="CamVid|StreetHazard|BddAnomaly")
    parser.add_argument("--seed",            type=int,      default=4040,       help="seed, if -1 no seed is use")
    parser.add_argument("--bsize",           type=int,      default=8,          help="batch size")
    parser.add_argument("--lr",              type=float,    default=0.2,        help="learning rate")  
    parser.add_argument("--temp",            type=float,    default=1.2,        help="temperature scaling ratio")
    parser.add_argument("--epoch",           type=int,      default=50,         help="number of epoch")
    parser.add_argument("--num_workers",     type=int,      default=4,          help="number of workers")              
    parser.add_argument("--num_nodes",       type=int,      default=1,          help="number of node")
    parser.add_argument("--optim",           type=str,      default="SGD",      help="type of optimizer SGD|AdamW")

    parser.add_argument("--drop",               action='store_true',            help="activate dropout in segnet")
    parser.add_argument("--wandb",              action='store_true',            help="activate wandb log")
    parser.add_argument("--resume",             action='store_true',            help="restart the training")
    parser.add_argument("--test",               action='store_true',            help="evaluate methods")

    args = parser.parse_args()

    # Setting multi GPUs
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = argsconfig(args)
    
    args.criterion = nn.CrossEntropyLoss()

    #### Preprocessing ####
    date_id = "{:%Y%m%d@%H%M%S}".format(datetime.datetime.now())
    args.dset_folder = "Datasets/" + args.data + "/"
    if not args.test:
        args.segnet_file = "logs/segnet_" + args.data + "_" + args.model + "_" + date_id + "/"
        os.mkdir(args.segnet_file)
    
    if args.wandb:
        wandb_name = date_id + "-" + args.data + "-" + args.model    
        wandb.init(project="Observer Network - SegNet", name = wandb_name, config = args, id = date_id)
            
    #### Print Args ####
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print()

    main(args)
    
