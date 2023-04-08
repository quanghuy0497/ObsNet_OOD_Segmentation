# Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation
Victor Besnier, Andrei Bursuc, David Picard & Alexandre Briot 

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2021

[Our paper](https://arxiv.org/abs/2108.01634)

## Abstract
In this paper, we propose a new method, named Observer Network, for OOD and error detection for semantic segmentation. 
We separate the segmentation and the error prediction by using a dedicated network to the later task, keeping the segmentation network unchanged.
We trigger failures of the Segmentation Network by applying Local Adversarial Attacks (LAA) on the input image during training. These images serve as proxy OOD to train the observer.  
We show that our method is fast, accurate and memory efficient on three different datasets and compare against multiple baselines.

##  Repository Structure
    ├ Obsnet/
    |    ├── Models/                                <- networks
    |    |      ├── road_anomaly_networks/          <- networks from SegmentMeIfYouCan
    |    |      ├── Segmenter/                      <- networks from Segmenter
    |    |      ├── load_net.py
    |    |      ├── deeplab_v3plus.py
    |    |      ├── resnet.py
    |    |      ├── obsnet.py
    |    |      └── segnet.py  
    |    |    
    |    ├── Dataset/                               <- loading  data
    |    |      ├── BddAnomaly/                     <- BDDAnomaly dataset
    |    |      ├── CamVid/                         <- CamVid dataset
    |    |      ├── StreetHazards/                  <- StreedHazards dataset
    |    |      ├── BDD_anomaly.py                  <- Dataset processing for BDD     
    |    |      ├── camvid.py                       <- Dataset Processing for CamVid     
    |    |      ├── cityscapes.py                   <- Dataset Processing for CityScapes     
    |    |      ├── load_data.py                    <- Dataloader   
    |    |      ├── seg_transfo.py                  <- Data augmentation for segmentation     
    |    |      └── street_hazard.py                <- Dataset Processing for StreetHazards  
    |    |
    |    ├── Utils/                                 <- useful fct
    |    |      ├── adv_attack.py                   <- fct adversarial attacks      
    |    |      ├── affichage.py                    <- fct for plot viridis & segment map       
    |    |      ├── loss.py                         <- focal loss      
    |    |      ├── metrics.py                      <- metrics for evaluation     
    |    |      └── utils.py                        <- useful functions
    |    ├── logs/                                  <- trained logs
    |    ├── obsnet_file/                           <- store pre-trained ObsNet models
    |    ├── segnet_file/                           <- store pre-trained SegNet models
    |    ├── configs.py                             <- Dataset params configs
    |    ├── engine_obsnet.py                       <- Training and evaluation code for obsnet
    |    ├── engine_segnet.py                       <- Training and evaluation code for segnet
    |    ├── main_obsnet.py                         <- Main code for obsnet
    |    ├── main_segnet.py                         <- Main code for segnet
    |    ├── inference.py                           <- perform inference on custom images
    |    ├── README.md                              <- me :)

## Usage
    
    $ git clone https://github.com/valeoai/obsnet
    $ cd obsnet 
    $ conda env create --file requirements.yml  
    $ conda activate obsnet
      
## Datasets

### CamVid
CamVid Dataset can be download here: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

The CamvidOOD dataset is composed of the same number of images as the classic CamVid. However, each image of the testing set contains an out-of-distribution animal.
You can download the CamVidOOD split in the release "CamVid OOD". Once CamVid and CamVidOOD are downloaded, execute the following line in your shell:
    
    $ cd /path/where/you/download/camvidood/
    $ unzip CamVidOOD.zip
    $ mv -r /path/where/you/download/camvidood/test_ood/ /path/to/dataset/CamVid/
    $ mv -r /path/where/you/download/camvidood/test_ood.txt /path/to/dataset/CamVid/
    $ mv -r /path/where/you/download/camvidood/testannot_ood/ /path/to/dataset/CamVid/
    
Folder Structure:

    ├ CamVid/
    |    ├ test/
    |    ├ testannot/
    |    ├ test_ood/
    |    ├ testannot_ood/
    |    ├ train/
    |    ├ trainannot/
    |    ├ val/
    |    ├ valannot/
    |    ├ test.txt
    |    ├ test_ood.txt
    |    ├ train.txt
    |    └ val.txt

### StreetHazards
Dataset can be download here: https://github.com/hendrycks/anomaly-seg

Folder Structure:

    ├ StreetHazards
    |    ├ annotations/
    |    |    ├ test/
    |    |    ├ training/
    |    |    └ validation/
    |    ├ images/
    |    |    ├ test/
    |    |    ├ training/
    |    |    └ validation/
    |    ├ README.txt
    |    ├ test.odgt
    |    ├ train.odgt
    |    └ validation.odgt

### BDD Anomaly
Dataset can be download here: https://github.com/hendrycks/anomaly-seg

Folder Structure:

    ├ BDD
    |    ├ bdd100k
    |    |    ├ seg
    |    |    |    ├ color_labels
    |    |    |    ├ images
    |    |    |    └ labels
    |    ├ test.odgt
    |    ├ train.odgt
    |    └ valiadation.odgt

## Training/Testing
According to the paper, the SegNet must be trained in prior to obtain the weight model. Then, the trained (SegNet) model is then freezed (as pre-trained model) feeding to Obsnet to training the OOD segmentation alone.

Unlike the original repository, we provide both SegNet and ObsNet training scheme, supporting CNN-based architectures (SegNet, DeepLabV3+) and ViT-based architecture (Segmenter). The first thing to do is training SegNet model first, then using its trained weight model for training ObsNet

### SegNet

+ Training:
    ```
    python main_segnet.py --dataset <dataset name> --model <model name> --wandb <this flag is optional>
    ```
    + The segnet trained weight model is automatically generated in `logs/segnet_<dataset_name>_<model_name>_<datetime>`, for example: `segnet_StreetHazards_segnet_20230408@083545`

+ Testing is not support for this scheme (currently), due to the presence of OOD categories in test set.

+ After training, please move the SegNet trained model, (i.e, `best.pth`) to folder `segnet_file\` for ObsNet training

### ObsNet

+ Training (including testing in the end):

    ```
      python main.py --model <model name> --data <dataset name> --adv <type of adversarial attack> --segnet_file <segnet_file name> --no_pretrained <optional> --wandb <optional>
    ```
    + `--no_pretrained`: Initialize the weight of the observer network with those of the segnet
          model
    + `--wandb`: log in WandB server
    + The obsnet trained weight model is automatically generated in `logs/obsnet_<dataset_name>_<model_name>_<datetime>`, for example: `obsnet_StreetHazards_segnet_20230408@083545`
    + After training, please move the ObsNet trained model to folder `obsnet_file\`

+ Testing only:
    ```
      python main.py --model <model name> --data <dataset name> --adv <type of adversarial attack> --segnet_file <segnet_file name> --obsnet_file <obsnet_file name> --test
    ```
    + Require both `--segnet_file` and `--obsnet_file` to have the model file name. By default, the 2 models is automatically selected in `segnet_file/` and `obsnet_file/`

### Inference 

You can perfom inference on a batch of images. 

Example:

    python inference.py --img_folder "<path to image folder>" --segnet_file "path to segnet" --obsnet_file "path to obsnet" --data "CityScapes" --model "raod_anomaly"    

## Pretrained ObsNet Models

Pretrained model for ObsNet are available in this [google drive](https://drive.google.com/drive/folders/11S9oK-Bk9PoP2728ldrROQKAa8Q3iUxx?usp=sharing). Please save on `obsnet_file/`

## Citation
If you find this repository usefull, please consider citing our [paper](https://arxiv.org/abs/2108.01634):

    @incollection{besnier2021trigger,
       title = {Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation},
       author = {Besnier, Victor and Bursuc, Andrei and Picard, David and Briot Alexandre},
       booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
       year = {2021}
       url= {https://arxiv.org/abs/2108.01634}
    }
