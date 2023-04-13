import pdb
import random

import torch
import numpy as np

import wandb
from Utils.affichage import plot
from Utils.utils import transform_output


def evaluate(epoch, segnet, loader, split, metric, args):
    
    segnet.eval()
    total_correct, total, avg_loss = 0, 0, 0.
    metric.reset()
    count = 0
    r = random.randint(0, len(loader) - 1)
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            bsize, channel, width, height = images.size()
            images = images.to(args.device)
            target = target.to(args.device)
            
            segnet_feat = segnet(images)                         
            
            segnet_pred = transform_output(pred=segnet_feat, bsize=bsize, nclass=args.nclass)

            loss = args.criterion(segnet_pred, target.view(-1))

            avg_loss += loss
            
            pred = segnet_pred.detach().max(1)[1]
            total_correct += pred.eq(target.view_as(pred)).sum()
            
            total += len(target.view(-1))
            metric.add(segnet_feat.detach(), target.view(-1, width, height))

            if args.test:
                if (i %20 == 0) and (args.wandb):
                    count += 1
                    seg_map = wandb.Image(plot(images, segnet_feat, target, args), caption="Segmentation map")
                    wandb.log({"Test Segmentation Map": seg_map}, step = count)
            else:
                if (i == r) and (args.wandb):
                    seg_map = wandb.Image(plot(images, segnet_feat, target, args), caption="Segmentation map")
                    wandb.log({"Val Segmentation Map": seg_map}, step = epoch + 1)
            
            print(f"\rEval loss: {loss.cpu().item():.4f}, "
                    f"Progress: {100 * (i / len(loader)):.2f}%", end="")
             

    avg_loss /= len(loader)
    
    Loss =  avg_loss.detach().cpu().item()
    Global_Avg = round(float(total_correct)/total * 100, 4)
    Class_Avg = round(metric.value()[2] * 100, 4)
    IoU = metric.value()[1] * 100

    print(f"\rEpoch {split} Summary: Train loss: {Loss:.4f}, "
          f"SegNet Acc: {Global_Avg:.2f}, "
          f"Class Acc: {Class_Avg:.2f}, "
          f"IoU: {IoU:.2f}")

    return Loss, Global_Avg, Class_Avg, IoU



def train(epoch, segnet, train_loader, optimizer, args):

    segnet.train()
    avg_loss, nb_sample, segnet_acc = 0, 0, 0.
    for i, (images, target) in enumerate(train_loader):
        bsize, channel, width, height = images.size()
        nb_sample += bsize * width * height

        images = images.to(args.device)
        target = target.to(args.device)

        segnet_feat = segnet(images)                         
        
        segnet_pred = transform_output(pred=segnet_feat, bsize=bsize, nclass=args.nclass)

        loss = args.criterion(segnet_pred, target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.cpu().item()
        
        segnet_pred = torch.argmax(segnet_pred, axis = 1).view(-1)
        
        segnet_acc += torch.sum(segnet_pred == target.view(-1))
        
        print(f"\rTrain loss: {loss.cpu().item():.4f}, "
              f"Progress: {100*(i/len(train_loader)):.2f}%", end="")

        if i == 0 and args.wandb:
            seg_map = wandb.Image(plot(images, segnet_feat, target, args), caption="Segmentation map")
            wandb.log({"Train Segmentation Map": seg_map}, step = epoch + 1)

    avg_loss /= len(train_loader)

    segnet_acc = 100 * (segnet_acc / nb_sample)

    print(f"\rEpoch Train Summary: Train loss: {avg_loss:.4f}, "
          f"SegNet Acc: {segnet_acc:.2f}")

    return avg_loss, segnet_acc
