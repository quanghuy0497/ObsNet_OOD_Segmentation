### The BDD Anomaly dataset:
Derived from [BDD100k dataset](https://bdd-data.berkeley.edu/portal.html), we follow the split from the paper [*Scaling Out-of-Distribution Detection for Real-World Settings*](https://arxiv.org/pdf/1911.11132.pdf) with the processing code from their [repository](https://github.com/hendrycks/anomaly-seg).

The original dataset consists of 7,000 training images and 1,000 validating images with 18 original classes. Follow the aforementioned paper, 3 classes *motocycle*, *train*, and *bicycle* are choosed as OOD objects and removed from the training and validationg dataset. Thus, the processed dataset consists of:
+ Training: 6,280 images
+ Validation: 910 images
+ Testing: 810 images  

### Download guide:

+ Download the 10k image set from [here](https://bdd-data.berkeley.edu/portal.html). Then store in `Datasets/BddAnomaly/seg/images/`. Remove the `test` dir
+ If necessery, download the annotate file and data processing code from [here](https://github.com/hendrycks/anomaly-seg), move it to `Datasets/BddAnomaly`. But these part are already stored in this repo
+ Run `create_bdd_dataset.py` to obtain the `train/test/valiatiln.odgt`

The folder structure should be:
    
    ├ BDDAnomaly
    |    ├ seg
    |    |    ├ color_labels
    |    |    ├ images
    |    |    |    ├ train
    |    |    |    └ val
    |    |    ├ labels
    |    |    └ train_labels
    |    ├ test.odgt
    |    ├ train.odgt
    |    └ valiadation.odgt

### Citation

    @article{hendrycks2019anomalyseg,
      title={Scaling Out-of-Distribution Detection for Real-World Settings},
      author={Hendrycks, Dan and Basart, Steven and Mazeika, Mantas and Zou, Andy and Kwon, Joe and Mostajabi, Mohammadreza and Steinhardt, Jacob and Song, Dawn},
      journal={ICML},
      year={2022}
    }

    @article{yu2018bdd100k,
    title={Bdd100k: A diverse driving video database with scalable annotation tooling},
    author={Yu, Fisher and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Liao, Mike and Madhavan, Vashisht and Darrell, Trevor},
    journal={arXiv preprint arXiv:1805.04687},
    volume={2},
    number={5},
    pages={6},
    year={2018}
    }
