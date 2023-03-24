import glob
import time
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as tf

from Models.segnet import SegNet
from Models.obsnet import Obsnet_Seg as ObsNet
from Models.deeplab_v3plus import deeplab_v3plus
from Models.road_anomaly_networks.deepv3 import DeepWV3Plus, DeepWV3Plus_Obsnet
from configs import argsconfig


def img2tensor(file, args):
    img = Image.open(file)
    img = img.resize((args.w, args.h), Image.BILINEAR)
    img = tf.to_tensor(img)
    img = tf.normalize(img, args.mean, args.std, False)
    img = img.unsqueeze(0).to(args.device)
    return img


def test(args):

    if args.model == "segnet":
        segnet = SegNet(3, args.nclass, init_vgg=False).to(args.device)
        obsnet = ObsNet(input_channels=3, output_channels=1).to(args.device)

    elif args.model == "deeplabv3plus":
        segnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True).to(args.device)
        obsnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True, obsnet=True).to(args.device)

    elif args.model == "road_anomaly":
        segnet = DeepWV3Plus(args.nclass).to(args.device)
        obsnet = DeepWV3Plus_Obsnet(num_classes=1).to(args.device)
    else:
        raise NameError("type of model not understood")

    segnet.load_state_dict(torch.load(args.segnet_file))
    segnet.eval()

    obsnet.load_state_dict(torch.load(args.obsnet_file))
    obsnet.eval()

    total_time = 0
    with torch.no_grad():
        for i, file in enumerate(args.imgs):
            img = img2tensor(file, args)

            # Inference
            start = time.time()
            _seg_pred, _obs_pred = _inference(img, segnet, obsnet)
            total_time += time.time() - start
            img = (img - img.min()) / (img.max() - img.min())

            # Post processing
            _seg_pred = torch.argmax(_seg_pred, dim=1)
            seg_pred = Image.fromarray(_seg_pred[0].byte().cpu().numpy()).resize((args.w, args.h))
            seg_pred.putpalette(args.colors.astype("uint8"))
            seg_pred = Image.blend(transforms.ToPILImage()(img[0]), seg_pred.convert("RGB"), alpha=0.5)

            _obs_pred = torch.sigmoid(_obs_pred)
            _obs_pred = (_obs_pred - _obs_pred.min()) / (_obs_pred.max() - _obs_pred.min())
            obs_pred = torch.zeros_like(_obs_pred)

            # highlight uncertainty for predicted instance
            for c in args.stuff_classes:
                mask = torch.where(_seg_pred[0] == c, args.one, args.zero)
                obs_pred += mask * _obs_pred
            obs_pred += 0.01 * _obs_pred
            obs_pred = torch.clamp(obs_pred, 0, 0.99)
            obs_pred = obs_pred * args.yellow + (1 - obs_pred) * args.blue

            obs_pred = Image.blend(transforms.ToPILImage()(img[0]), transforms.ToPILImage()(obs_pred[0]), alpha=0.9)

            # Concat and Save visualization
            res = Image.new('RGB', (3 * args.w, args.h))
            res.paste(transforms.ToPILImage()(img[0]), (0, 0))
            res.paste(seg_pred, (args.w, 0))
            res.paste(obs_pred, (2 * args.w, 0))
            res.save(f'./save_img/{file.split("/")[-1]}')

            print(f"Test: image: {i}, progression: {i/len(args.imgs)*100:.1f} %, img = {file.split('/')[-1]} ")
    print(f"Inference run time: {len(args.imgs)/total_time:.1f} FPS")


def _inference(img, segnet, obsnet):
    seg_feat = segnet(img, return_feat=True)
    obs_pred = torch.sigmoid(obsnet(img, seg_feat)).squeeze()
    return seg_feat[-1], obs_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",            type=str,      default="",         help="Type of dataset")
    parser.add_argument("--model",           type=str,      default="",         help="Segnet|deeplabv3plus|road_anomaly")
    parser.add_argument("--img_folder",      type=str,      default="",         help="path to image folder")
    parser.add_argument("--segnet_file",     type=str,      default="",         help="path to segnet")
    parser.add_argument("--obsnet_file",     type=str,      default="",         help="path to obsnet")
    parser.add_argument("--seed",            type=int,      default=4040,       help="seed, if -1 no seed is use")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = argconfigs(args)
    

    args.imgs = glob.glob(args.img_folder + "*")
    args.cmap = dict(zip(range(len(args.colors)), args.colors))
    args.yellow = torch.FloatTensor([1, 1, 0]).to(args.device).view(1, 3, 1, 1).expand(1, 3, args.h, args.w)
    args.blue = torch.FloatTensor([0, 0, .4]).to(args.device).view(1, 3, 1, 1).expand(1, 3, args.h, args.w)

    test(args)

