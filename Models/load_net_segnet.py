import os
import torch
import torch.nn as nn
from Models.segnet import SegNet
from Models.deeplab_v3plus import deeplab_v3plus
from Models.Segmenter.factory import load_segmenter

from Models.road_anomaly_networks.deepv3 import DeepWV3Plus


def net_loader(args):
    """ load the observer network and the segmentation network """

    if args.model == "segnet":
        segnet = SegNet(3, args.nclass, init_vgg=True).to(args.device)

    elif args.model == "deeplabv3plus":
        segnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True).to(args.device)
    elif args.model == "road_anomaly":
        segnet = DeepWVclear3Plus(args.nclass).to(args.device)
    elif args.model == "segmenter":
        segnet = load_segmenter(num_classes = args.nclass, obsnet=False, img_size = args.size).to(args.device)

    else:
        raise NameError("Model not known")

        
    if args.test or args.resume:
        segnet.load_state_dict(torch.load(os.path.join(args.segnet_file, "best.pth")))

    if not args.test:
        segnet = nn.DataParallel(segnet) #, device_ids = [1])
        
    return segnet
