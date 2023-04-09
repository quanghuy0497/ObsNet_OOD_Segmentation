import os
import torch
import torch.nn as nn
from Models.segnet import SegNet
from Models.obsnet import Obsnet_Seg, Obsnet_Small
from Models.deeplab_v3plus import deeplab_v3plus
from Models.Segmenter.factory import load_segmenter

from Models.road_anomaly_networks.deepv3 import DeepWV3Plus, DeepWV3Plus_Obsnet


def net_loader(args):
    """ load the observer network and the segmentation network """

    if args.model == "segnet":
        segnet = SegNet(3, args.nclass, init_vgg=False).to(args.device)
        if args.obs_mlp:
            obsnet = Obsnet_Small(input_channels=512, output_channels=1).to(args.device)
        else:
            obsnet = Obsnet_Seg(input_channels=3, output_channels=1).to(args.device)

    elif args.model == "deeplabv3plus":
        segnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True).to(args.device)
        obsnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True, obsnet=True).to(args.device)

    elif args.model == "road_anomaly":
        segnet = DeepWVclear3Plus(args.nclass).to(args.device)
        obsnet = DeepWV3Plus_Obsnet(num_classes=1).to(args.device)
    
    elif args.model == "segmenter":
        segnet = load_segmenter(num_classes = args.nclass, obsnet=False, img_size = args.size).to(args.device)
        obsnet = load_segmenter(num_classes = args.nclass, obsnet=True, img_size = args.size).to(args.device)

    else:
        raise NameError("Model not known")
    
    print(f"Load SegNet file: {args.segnet_file}")

    if args.model == "segmenter":
        segnet.load_state_dict(torch.load(args.segnet_file)['model'], strict=True)
    else:
        segnet.load_state_dict(torch.load(args.segnet_file))
        
    segnet.eval()

    if args.test or args.resume:
        if args.obsnet_file:
            print(f"Load ObsNet file: {args.obsnet_file}")
            obsnet.load_state_dict(torch.load(args.obsnet_file))
        else:
            obsnet_file = os.path.join(args.log, "best.pth")
            print(f"Load ObsNet file: {obsnet_file}")
            obsnet.load_state_dict(torch.load(obsnet_file))

    if args.test:
        obsnet.eval()

    if not args.test:
        segnet = nn.DataParallel(segnet, device_ids = [0])
        obsnet = nn.DataParallel(obsnet, device_ids = [0])

    return obsnet, segnet
