import pdb
import random

import torch
from torchvision.utils import make_grid

import wandb
from Utils.adv_attacks import select_attack
from Utils.affichage import draw, plot
from Utils.metrics import print_result
from Utils.utils import (ensemble_estimation, entropy, gauss_estimation,
                         mcda_estimation, monte_carlo_estimation,
                         odin_estimation, transform_output)


def evaluate(epoch, obsnet, segnet, loader, split, args):
    """ Evaluate method contain in the arguments args.test_multi
        epoch  ->  int: current epoch
        split  ->  str: Test or Val
        loader ->  torch.DataLoader: the dataloader
        obsnet ->  torch.Module: the observer to test
        segnet ->  torch.module: the segnet pretrained and freeze
        args   ->  Argparse: global parameters
    return:
        avg_loss     -> float: average loss on the dataset
        results_obs  -> dict: the result of the obsnet on different metrics
    """

    obsnet.eval()
    segnet.eval()
    avg_loss, nb_sample, obsnet_acc, segnet_acc = 0, 0, 0., 0.
    r = random.randint(0, len(loader) - 1)
    count = 0
    for i, (images, target) in enumerate(loader):
        bsize, channel, width, height = images.size()
        nb_sample += bsize * width * height

        images = images.to(args.device)
        target = target.to(args.device)

        with torch.no_grad():
            segnet_feat = segnet(images, return_feat=True)                                  # SegNet forward
            segnet_logit = transform_output(pred=segnet_feat[-1], bsize=bsize, nclass=args.nclass)

            pred = segnet_logit.max(-1)[1]

            supervision = pred.view(-1) != target.view(-1)                                  # GT for ObsNet Training
            supervision = torch.where(supervision, args.one, args.zero).to(args.device)
            supervision = supervision.view(bsize, -1)

            pred = pred.float().view(-1, 1)
            target = target.float().view(-1, 1)

            if "mc_drop" in args.test_multi:  # Entropy of T forward passes with dropout
                mc = monte_carlo_estimation(segnet, images, bsize, width, height, args)
                mc = entropy(mc, softmax=False).view(-1, 1)
                batch_res_mc = torch.cat((mc, pred, target), dim=1)
                res_mc = batch_res_mc.cpu() if i == 0 else torch.cat((res_mc, batch_res_mc.cpu()), dim=0)

            if "mcp" in args.test_multi:  # 1 minus the max of the prediction
                sm = 1 - torch.softmax(segnet_logit, -1).max(-1)[0]
                batch_res_sm = torch.cat((sm.view(-1, 1), pred, target), dim=1)
                res_sm = batch_res_sm.cpu() if i == 0 else torch.cat((res_sm, batch_res_sm.cpu()), dim=0)

            if "temp_scale" in args.test_multi:  # 1 minus the max of the temperated prediction
                ts = 1 - torch.softmax(segnet_logit/args.temp, -1).max(-1)[0]
                batch_res_ts = torch.cat((ts.view(-1, 1), pred, target), dim=1)
                res_ts = batch_res_ts.cpu() if i == 0 else torch.cat((res_ts, batch_res_ts.cpu()), dim=0)

            if "void" in args.test_multi:  # probability of the class void/void
                void = segnet_logit[:, 3] if args.data == "StreetHazard" else segnet_logit[:, -1]
                batch_res_void = torch.cat((void.view(-1, 1), pred, target), dim=1)
                res_void = batch_res_void.cpu() if i == 0 else torch.cat((res_void, batch_res_void.cpu()), dim=0)

            if "gauss" in args.test_multi:  # entropy of T forward passes with noise on the parameters
                gauss = gauss_estimation(segnet, images, bsize, width, height, args)
                gauss = entropy(gauss, softmax=False).view(-1, 1)
                batch_res_gauss = torch.cat((gauss.float().view(-1, 1), pred, target), dim=1)
                res_gauss = batch_res_gauss.cpu() if i == 0 else torch.cat((res_gauss, batch_res_gauss.cpu()), dim=0)

            if "mcda" in args.test_multi:  # Entropy of T forward passes with data augmentation
                mcda = mcda_estimation(segnet, images, bsize, width, height, args)
                mcda = entropy(mcda, softmax=False).view(-1, 1)
                batch_res_mcda = torch.cat((mcda, pred, target), dim=1)
                res_mcda = batch_res_mcda.cpu() if i == 0 else torch.cat((res_mcda, batch_res_mcda.cpu()), dim=0)

            if "ensemble" in args.test_multi:  # Entropy of M differents networks's prediction
                ens = ensemble_estimation(segnet, images, bsize, width, height, args)
                ens_entrop = entropy(ens, softmax=False).view(-1, 1)
                batch_res_ens = torch.cat((ens_entrop, ens.max(-1)[1].float().view(-1, 1), target), dim=1)
                res_ens = batch_res_ens.cpu() if i == 0 else torch.cat((res_ens, batch_res_ens.cpu()), dim=0)

            if "obsnet" in args.test_multi:  # our method
                obs_pred = obsnet(images, segnet_feat, no_residual=args.no_residual, no_img=args.no_img)
                obs_pred = transform_output(pred=obs_pred, bsize=bsize, nclass=1)
                loss = args.criterion(obs_pred.view(-1), supervision.view(-1))

                segnet_acc += pred.view(-1).eq(target.view(-1)).sum()
                obsnet_acc += torch.round(torch.sigmoid(obs_pred)).view(-1).eq(supervision.view(-1)).sum()
                avg_loss += loss.cpu().item()

                obs = torch.sigmoid(obs_pred)
                batch_res_obs = torch.cat((obs.view(-1, 1), pred, target), dim=1)
                res_obs = batch_res_obs.cpu() if i == 0 else torch.cat((res_obs, batch_res_obs.cpu()), dim=0)

                print(f"\rEval loss: {loss.cpu().item():.4f}, "
                      f"Progress: {100 * (i / len(loader)):.2f}%",
                      end="")
                if args.test:
                    if i % 20 == 0 and args.wandb:      # save several images simultaneously
                        sm = 1 - torch.softmax(segnet_feat[-1], 1).max(1)[0][0]     
                        sm_uncertainty = draw(sm, args).cpu()

                        obs_pred = obs_pred.view(bsize, -1)     # ObsNet visualization
                        obsnet_uncertainty = draw(torch.sigmoid(obs_pred[0]), args).cpu()

                        obs_label = supervision.view(bsize, -1) # GT visualization
                        label = draw(obs_label[0], args).cpu()

                        uncertainty_map = torch.cat((obsnet_uncertainty, sm_uncertainty, label), dim=0)
                        uncertainty_map = make_grid(uncertainty_map, normalize=False)

                        seg_map = wandb.Image(plot(images, segnet_feat[-1], target.view(bsize, -1), args=args), caption='Segmentation Map')
                        ood_map = wandb.Image(uncertainty_map, caption="Uncertainty map")
                        count += 1
                        wandb.log({"Test/Segmentation Map": seg_map, "Test/Uncertainty Map": ood_map}, step = count)
                else:
                    if i == r and args.wandb:             # save 1 image randomly          
                        sm = 1 - torch.softmax(segnet_feat[-1], 1).max(1)[0][0]         
                        sm_uncertainty = draw(sm, args).cpu()

                        obs_pred = obs_pred.view(bsize, -1)     # ObsNet visualization
                        obsnet_uncertainty = draw(torch.sigmoid(obs_pred[0]), args).cpu()

                        obs_label = supervision.view(bsize, -1) # GT visualization
                        label = draw(obs_label[0], args).cpu()

                        uncertainty_map = torch.cat((obsnet_uncertainty, sm_uncertainty, label), dim=0)
                        uncertainty_map = make_grid(uncertainty_map, normalize=False)

                        seg_map = wandb.Image(plot(images, segnet_feat[-1], target.view(bsize, -1), args=args), caption='Segmentation Map')
                        ood_map = wandb.Image(uncertainty_map, caption="Uncertainty map")
                        wandb.log({"Val/Segmentation Map": seg_map, "Val/Uncertainty Map": ood_map}, step = epoch + 1)

        if "odin" in args.test_multi:                     # output of ODIN
            odin = odin_estimation(segnet, images, bsize, args)
            batch_res_odin = torch.cat((odin.view(-1, 1), pred, target), dim=1)
            res_odin = batch_res_odin.cpu() if i == 0 else torch.cat((res_odin, batch_res_odin.cpu()), dim=0)

    if "mcp" in args.test_multi:        print_result("Softmax", split, res_sm, epoch, args)
    if "temp_scale" in args.test_multi: print_result("Temp scale", split, res_ts, epoch, args)
    if "void" in args.test_multi:       print_result("Void", split, res_void, epoch, args)
    if "mcda" in args.test_multi:       print_result("MCDA", split, res_mcda, epoch, args)
    if "gauss" in args.test_multi:      print_result("Gauss", split, res_gauss, epoch, args)
    if "ensemble" in args.test_multi:   print_result("Ensemble", split, res_ens, epoch, args)
    if "mc_drop" in args.test_multi:    print_result("MC Dropout", split, res_mc, epoch, args)
    if "odin" in args.test_multi:       print_result("Odin", split, res_odin, epoch, args)
    if  args.test:
        res_obs = print_result("ObsNet", "Test", res_obs, epoch, args)
    else:
        res_obs = print_result("ObsNet", "Val", res_obs, epoch, args)

    avg_loss /= len(loader)
    obsnet_acc = 100 * (obsnet_acc / nb_sample)
    segnet_acc = 100 * (segnet_acc / nb_sample)

    print(f"\rEpoch {split} Summary: Avg loss: {avg_loss:.4f}, "
          f"ObsNet acc: {obsnet_acc:.2f}, "
          f"SegNet acc: {segnet_acc:.2f}\t"
          )

    return [obsnet_acc, segnet_acc, avg_loss, res_obs] if "obsnet" in args.test_multi else avg_loss



def train(epoch, obsnet, segnet, train_loader, optimizer, args):
    """ Train the observer network for one epoch
        epoch        ->  int: current epoch
        obsnet       ->  torch.Module: the observer to train
        segnet       ->  torch.Module: the segnet pretrained and freeze
        train_loader ->  torch.DataLoader: the training dataloader
        optimizer    ->  torch.optim: optimizer to train observer
        args         ->  Argparse: global parameters
    return:
        avg_loss -> float : average loss on the dataset
    """

    obsnet.train()
    segnet.eval()
    avg_loss, nb_sample, obsnet_acc, segnet_acc = 0, 0, 0., 0.
    for i, (images, target) in enumerate(train_loader):
        bsize, channel, width, height = images.size()
        nb_sample += bsize * width * height

        images = images.to(args.device)
        target = target.to(args.device)

        if args.adv != "none":                                                             # perform the LAA
            images, mask = select_attack(images, target, segnet, args)

        with torch.no_grad():
            segnet_feat = segnet(images, return_feat=True)                                 # SegNet forward
            segnet_pred = transform_output(pred=segnet_feat[-1], bsize=bsize, nclass=args.nclass)
            segnet_pred = torch.argmax(segnet_pred, axis = 1).view(-1)
            
            error = segnet_pred != target.view(-1)             # GT for observer training
            supervision = torch.where(error, args.one, args.zero).to(args.device).view(-1)

        obs_pred = obsnet(images, segnet_feat, no_residual=args.no_residual, no_img=args.no_img)
        obs_pred = transform_output(pred=obs_pred, bsize=bsize, nclass=1)

        loss = args.criterion(obs_pred.view(-1), supervision)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.cpu().item()
        
        segnet_acc += torch.sum(segnet_pred == target.view(-1))
        obsnet_acc += torch.sum(torch.round(torch.sigmoid(obs_pred)).view(-1) == supervision)

        print(f"\rTrain loss: {loss.cpu().item():.4f}, "
              f"Progress: {100*(i/len(train_loader)):.2f}%",
              end="")

        if i == 0:                                                                      # Visualization
            with torch.no_grad():
                sm = segnet_feat[-1]                                                    # MCP visualization
                sm = 1 - torch.softmax(sm, 1).max(1)[0][0]
                sm_uncertainty = draw(sm, args).cpu()

                obs_pred = obs_pred.view(bsize, -1)                                     # ObsNet visualization
                obsnet_uncertainty = draw(torch.sigmoid(obs_pred[0]), args).cpu()

                obs_label = supervision.view(bsize, -1)                                 # GT visualization
                label = draw(obs_label[0], args).cpu()

                uncertainty_map = torch.cat((obsnet_uncertainty, sm_uncertainty, label), dim=0)
                uncertainty_map = make_grid(uncertainty_map, normalize=False)
                segmentation_map = plot(images + (10 * mask), segnet_feat[-1], target, args)

            if args.wandb:
                ood_map = wandb.Image(uncertainty_map, caption="Uncertainty map")
                seg_map = wandb.Image(segmentation_map, caption="Segmentation map")
                wandb.log({"Train/Segmentation Map": seg_map, "Train/Uncertainty Map": ood_map}, step = epoch + 1)

    avg_loss /= len(train_loader)

    obsnet_acc = 100 * (obsnet_acc / nb_sample)
    segnet_acc = 100 * (segnet_acc / nb_sample)

    print(f"\rEpoch Train Summary: Train Avg loss: {avg_loss:.4f}, "
          f"ObsNet acc: {obsnet_acc:.2f}, "
          f"SegNet acc: {segnet_acc:.2f}"
          )

    return avg_loss, obsnet_acc, segnet_acc
