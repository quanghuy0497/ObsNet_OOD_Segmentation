import argparse
import datetime
import os
import pdb
import pprint
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler

import wandb
from configs import argsconfig
from Datasets.load_data import data_loader
from engine_obsnet import evaluate, train
from Models.load_net_obsnet import net_loader
from Utils.utils import reinitialize


def main(args):
    """ Manage the training, the testing and the declaration of the 'global parameters'
        such as optimizer, scheduler, the SumarryWriter, etc. """
        
    # Load Dataset
    train_loader, val_loader, test_loader = data_loader(args.data, args)

    args.cmap = train_loader.dataset.cmap
    args.class_name = train_loader.dataset.class_name
    args.colors = np.array(train_loader.dataset.colors)
    
    # Load Network
    obsnet, segnet = net_loader(args)
    
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(obsnet.parameters(), lr=args.lr)
    elif args.optim == "AdamW":
        optimizer = torch.optim.AdamW(obsnet.parameters(), lr=args.lr)

    sched = scheduler.MultiStepLR(optimizer, milestones=[args.epoch // 2, args.epoch-5], gamma=0.2)

    if args.test:
        test_obsnet_acc, test_segnet_acc, test_loss, test_results_obs = evaluate(0, obsnet, segnet, test_loader, "Test", args)
        if args.wandb:
            wandb.log({'Test/Loss': test_loss, 'Test/ObsNet Acc': test_obsnet_acc, 'Test/SegNet Acc': test_segnet_acc, 'Test/AuROC': test_results_obs["auroc"], 'Test/FPR@95': test_results_obs["fpr_at_95tpr"], 'Test/AuPR': test_results_obs["aupr"],'Test/ACE': test_results_obs["ace"]})
    else:
        if not args.no_pretrained:
            reinitialize(obsnet, args.segnet_file)
        best = 0
        start = time.time()
        
        print("===================================")
        print("Training phase: ")
        print("===================================")
        
        for epoch in range(0, args.epoch+1):
            print(f"\n######## Epoch: {epoch} || Time: {(time.time() - start)/60:.2f} min ########")

            train_loss, train_obsnet_acc, train_segnet_acc = train(epoch, obsnet, segnet, train_loader, optimizer, args)
            val_obsnet_acc, val_segnet_acc, val_loss, train_results_obs = evaluate(epoch, obsnet, segnet, val_loader, "Val", args)
            
            
            if args.wandb:
                wandb.log({'Train/Loss': train_loss, 'Train/ObsNet Acc': train_obsnet_acc, 'Train/SegNet Acc': train_segnet_acc, 'Train/AuROC': train_results_obs["auroc"], 'Train/FPR@95': train_results_obs["fpr_at_95tpr"], 'Train/AuPR': train_results_obs["aupr"],'Train/ACE': train_results_obs["ace"]}, step = epoch + 1)
                
                wandb.log({'Val/Loss': val_loss, 'Val/ObsNet Acc': val_obsnet_acc, 'Val/SegNet Acc': val_segnet_acc}, step = epoch + 1)

            if epoch % 5 == 0:               # save Checkpoint
                model_to_save = obsnet.module.state_dict()
                torch.save(model_to_save, os.path.join(args.log, f"epoch{epoch:03d}.pth"))
            model_to_save = obsnet.module.state_dict()
            torch.save(model_to_save, os.path.join(args.log, f"checkpoint.pth"))

            if train_results_obs["auroc"] > best:  # Save Best model
                print("save best net!!!")
                best = train_results_obs["auroc"]
                model_to_save = obsnet.module.state_dict()
                torch.save(model_to_save, os.path.join(args.log, "best.pth"))
            sched.step()
        
        print("===================================")
        print("Testing phase: ")
        print("===================================")
        
        ### Load network again to obtain best trained obsnet network
        args.test = True
        obsnet, segnet = net_loader(args)                   
        
        test_obsnet_acc, test_segnet_acc, test_loss, test_results_obs = evaluate(0, obsnet, segnet, test_loader, "Test", args)
        if args.wandb:
            wandb.log({'Test/Loss': test_loss, 'Test/ObsNet Acc': test_obsnet_acc, 'Test/SegNet Acc': test_segnet_acc, 'Test/AuROC': test_results_obs["auroc"], 'Test/FPR@95': test_results_obs["fpr_at_95tpr"], 'Test/AuPR': test_results_obs["aupr"],'Test/ACE': test_results_obs["ace"]})
             
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    ### Argparse ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--segnet_file",     type=str,      default="",         help="path to segnet")
    parser.add_argument("--obsnet_file",     type=str,      default="",         help="path to obsnet")
    parser.add_argument("--log",             type=str,      default="",         help="obsnet models log")
    parser.add_argument("--model",           type=str,      default="segnet",   help="segnet|deeplabv3plus|road_anomaly|segmenter")
    parser.add_argument("--data",            type=str,      default="",         help="CamVid|StreetHazard|BddAnomaly")
    parser.add_argument("--t",               type=int,      default=50,         help="number of forward pass for ensemble")
    parser.add_argument("--seed",            type=int,      default=4040,       help="seed, if -1 no seed is use")
    parser.add_argument("--bsize",           type=int,      default=8,          help="batch size")
    parser.add_argument("--lr",              type=float,    default=0.2,        help="learning rate of obsnet")  
    parser.add_argument("--temp",            type=float,    default=1.2,        help="temperature scaling ratio")
    parser.add_argument("--epsilon",         type=float,    default=0.025,      help="epsilon for adversarial attacks")
    parser.add_argument("--gauss_lambda",    type=float,    default=0.002,      help="lambda parameters for gauss params")
    parser.add_argument("--epoch",           type=int,      default=50,         help="number of epoch")
    parser.add_argument("--num_workers",     type=int,      default=4,          help="number of workers")              
    parser.add_argument("--adv",             type=str,      default="none",     help="type of adversarial attacks")
    parser.add_argument("--optim",           type=str,      default="SGD",      help="type of optimizer SGD|AdamW")
    parser.add_argument("--test_multi",      type=str,      default="obsnet",   help="test all baseline, split by comma")
    parser.add_argument("--wandb",              action='store_true',            help="activate wandb log")
    parser.add_argument("--no_img",             action='store_true',            help="use image for obsnet")
    parser.add_argument("--resume",             action='store_true',            help="restart the training")
    parser.add_argument("--obs_mlp",            action='store_true',            help="use a smaller archi for obsnet")
    parser.add_argument("--test",               action='store_true',            help="evaluate methods")
    parser.add_argument("--no_residual",        action='store_true',            help="remove residual connection for obsnet")
    parser.add_argument("--no_pretrained",      action='store_true',            help="load segnet weight for the obsnet")
    args = parser.parse_args()

    # Setting multi GPUs
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = argsconfig(args)
    

    args.test_multi = args.test_multi.split(",")

    #### Preprocessing ####
    date_id = "{:%Y%m%d@%H%M%S}".format(datetime.datetime.now())
    args.dset_folder = "Datasets/" + args.data + "/"
    if not args.test:
        args.log = "logs/obsnet_" + args.data +  "_" + args.model + "_" + date_id + "/"
        os.mkdir(args.log)
    else:
        args.log = "logs/" + args.log
        if args.obsnet_file:
            args.obsnet_file = 'obsnet_file/' + args.obsnet_file
            
    if args.wandb:
        if args.no_pretrained:
            wandb_name = date_id + "-" + args.data + "-" + args.model + "-" + args.adv + "-no_pretrained"
        else:
            wandb_name = date_id + "-" + args.data + "-" + args.model  + "-" + args.adv + "-" + args.segnet_file
            
        wandb.init(project="Observer Network - ObsNet", name = wandb_name, config = args, id = date_id)
        
    args.segnet_file = 'segnet_file/' +  args.segnet_file
    
    #### Print Args ####
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print()

    main(args)
    
