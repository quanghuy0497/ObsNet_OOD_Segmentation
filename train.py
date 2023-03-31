import torch
import wandb
import pdb
from torchvision.utils import make_grid
from Utils.affichage import draw, plot
from Utils.utils import transform_output
from Utils.adv_attacks import select_attack


def training(epoch, obsnet, segnet, train_loader, optimizer, args):
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
