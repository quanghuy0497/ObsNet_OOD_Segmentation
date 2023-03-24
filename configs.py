import torch
import torch.nn as nn
import numpy as np
import random

def argsconfig(args):
    args.one = torch.FloatTensor([1.]).to(args.device)
    args.zero = torch.FloatTensor([0.]).to(args.device)
    
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(args.seed)

    if args.data == "CamVid":
        args.mean = [0.4108, 0.4240, 0.4316]
        args.std = [0.3066, 0.3111, 0.3068]
        args.h, args.w = [360, 480]
        args.size = [args.h, args.w]
        args.crop = (50, 80)
        args.pos_weight = torch.tensor([2]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [128, 128, 60, 80]
        args.nclass = 12
        args.colors = np.array([[128, 128, 128],        # sky
                                [128,   0,   0],        # building
                                [192, 192, 128],        # column_pole
                                [128,  64, 128],        # road
                                [  0,   0, 192],        # sidewalk
                                [128, 128,   0],        # Tree
                                [192, 128, 128],        # SignSymbol
                                [ 64,  64, 128],        # Fence
                                [ 64,   0, 128],        # Car
                                [ 64,  64,   0],        # Pedestrian
                                [  0, 128, 192],        # Bicyclist
                                [  0,   0,   0]])       # Void

        args.stuff_classes = [8, 9, 10]

    elif args.data == "BddAnomaly":
        args.h, args.w = [360, 640]  # Original size [720, 1280]
        args.size = [args.h, args.w] 
        args.crop = (80, 150)
        args.pos_weight = torch.tensor([3]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [300, 360, 160, 200]
        args.mean = [0.3698, 0.4145, 0.4247]
        args.std = [0.2525, 0.2695, 0.2870]
        args.nclass = 19
        args.colors = np.array([[128,  64, 128],        # road
                                [244,  35, 232],        # sidewalk
                                [ 70,  70,  70],        # building
                                [102, 102, 156],        # wall
                                [190, 153, 153],        # fence
                                [153, 153, 153],        # pole
                                [250, 170,  30],        # traffic_light
                                [220, 220,   0],        # traffic_sign
                                [107, 142,  35],        # vegetation
                                [152, 251, 152],        # terrain
                                [  0, 130, 180],        # sky
                                [220,  20,  60],        # person
                                [255,   0,   0],        # rider
                                [  0,   0, 142],        # car
                                [  0,   0,  70],        # truck
                                [  0,  60, 100],        # bus
                                [  0,  80, 100],        # train
                                [  0,   0, 230],        # motorcycle
                                [119,  11,  32],        # bicycle
                                [  0,   0,   0]])       # unlabelled
        
        args.stuff_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19]

    elif args.data == "CityScapes":
        args.h, args.w = [512, 1024]   # original size [1024, 2048]
        args.size = [args.h, args.w]
        args.crop = (150, 250)
        args.pos_weight = torch.tensor([3]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [128, 128, 60, 80]
        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.225]
        args.nclass = 19
        args.object_class = torch.LongTensor([12, 13, 14, 15, 16, 17, 18, 19]).to(args.device)
        args.colors = np.array([[128,  64, 128],        # 0: road
                                [244,  35, 232],        # 1: sidewalk
                                [ 70,  70,  70],        # 2: building
                                [102, 102, 156],        # 3: wall
                                [190, 153, 153],        # 4: fence
                                [153, 153, 153],        # 5: pole
                                [250, 170,  30],        # 6: traffic_light
                                [220, 220,   0],        # 7: traffic_sign
                                [107, 142,  35],        # 8: vegetation
                                [152, 251, 152],        # 9: terrain
                                [  0, 130, 180],        # 10: sky
                                [220,  20,  60],        # 11: person
                                [255,   0,   0],        # 12: rider
                                [  0,   0, 142],        # 13: car
                                [  0,   0,  70],        # 14: truck
                                [  0,  60, 100],        # 15: bus
                                [  0,  80, 100],        # 16: train
                                [  0,   0, 230],        # 17: motorcycle
                                [119,  11,  32],        # 18: bicycle
                                [  0,   0,   0]])       # 19: unlabelled

        args.stuff_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        
    elif args.data == "StreetHazards":
        args.h, args.w = [360, 640]  # Original size [720, 1280]
        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.225]
        args.nclass = 14
        args.size = [args.h, args.w]
        args.crop = (80, 150)
        args.pos_weight = torch.tensor([3]).to(args.device)
        args.criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight)
        args.patch_size = [300, 360, 160, 200]
        args.colors = np.array([[  0,   0,   0],        # unlabeled
                                [ 70,  70,  70],        # building
                                [190, 153, 153],        # fence
                                [250, 170, 160],        # other
                                [220,  20,  60],        # pedestrian
                                [153, 153, 153],        # pole
                                [157, 234,  50],        # road line
                                [128,  64, 128],        # road
                                [244,  35, 232],        # sidewalk
                                [107, 142,  35],        # vegetation
                                [  0,   0, 142],        # car
                                [102, 102, 156],        # wall
                                [220, 220,   0],        # traffic sign
                                [ 60, 250, 240]])       # anomaly
                                
        args.stuff_classes = [13]

    else:
        raise NameError("Data not known")
    
    return args