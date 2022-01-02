
#  Viewpoint Invariant Dense Matching for Visual Geolocalization: PyTorch implementation

<img src="https://github.com/gmberton/geo_warp/blob/main/images_paper/teaser.jpg" width="500">

This is the implementation of the ICCV21 paper:

G Berton, C. Masone, V. Paolicelli and B. Caputo, [Viewpoint Invariant Dense Matching for Visual Geolocalization](https://arxiv.org/pdf/2109.09827.pdf)

##  Setup

First download the baseline models which have been trained following the training procedure in the [NetVLAD paper](https://arxiv.org/abs/1511.07247).
We provide a script to download the six models used, which are a combination of 3 backbone encoders (AlexNet, VGG-16 and ResNet-50) with 2 pooling/aggregation layers (GeM and NetVLAD). The models are automatically saved in trained_networks/pretrained_baselines.

```bash

python download_pretrained_baselines.py

```

Then you should prepare your geo-localization dataset, so that the directory tree is as such:
```
dataset_name
└── images
    ├── train
    │   ├── gallery
    │   └── queries
    ├── val
    │   ├── gallery
    │   └── queries
    └── test
        ├── gallery
        └── queries
```
and the images are named as ```@UTM east@UTM north@whatever@.jpg```


###  Dependencies

See `requirements.txt`

##  Training

You can train the model using the `train.py`, here's an example with the lightest/fastest model (i.e. AlexNet + GeM):

```bash

python train.py --arch alexnet --pooling gem --resume_fe trained_networks/pretrained_baselines/alexnet_gem.pth

```

For a full set of options, and explanation of the parameters, run `python train.py -h`.
The script will create a folder under `./runs/default/YYYY-MM-DD_HH-mm-ss` where logs and checkpoints will be saved. At the end of the training you will see the results with the baseline model, as well as when re-ranking is applied using GeoWarp.

##  Evaluation

You can use this code to compute the results with our trained models. To reproduce the results from the paper, you can download our models simply running

```bash

python download_trained_hom_reg.py

```

which will automatically download the models and save them under trained_networks/trained_homography_regressions. Then to obtain the results you can execute

```bash

python eval.py --arch alexnet --pooling gem --resume_fe trained_networks/pretrained_baselines/alexnet_gem.pth --resume_hr trained_networks/trained_homography_regressions/alexnet_gem.pth

```

This will give you the exact same results as in Table 1 of the paper.
For a full set of options, and explanation of the parameters, run `python eval.py -h`.

###  BibTeX

If you use this code in your project, please cite us using:

```bibtex

@InProceedings{Berton_ICCV_2021,
    author    = {Berton, Gabriele and Masone, Carlo and Paolicelli, Valerio and Caputo, Barbara},
    title     = {Viewpoint Invariant Dense Matching for Visual Geolocalization},
    booktitle = ICCV,
    month     = {October},
    year      = {2021},
    pages     = {12169-12178}
}

```

