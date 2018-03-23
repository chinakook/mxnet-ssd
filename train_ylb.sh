#!/bin/bash

python tools/prepare_ylb.py --root /mnt/6B133E147DED759E/tmp/ylb --shuffle False --target ./data/train.lst

python train.py --val-path "" --voc07 False --gpus 1 --batch-size 32 --num-example 298 --num-class 1 --class-names "" --network "mobilenet" --pretrained "/home/dingkou/dev/ylb_det/model/mobilenet" --freeze "" --lr 0.0003 --end-epoch 600 --wd 0.00005

MXNET_CUDNN_AUTOTUNE_DEFAULT=0 python demo.py --epoch 108 --images /mnt/6B133E147DED759E/tmp/ylb/zp/IMG_20170728_103637.jpg --thresh 0.5 --network vgg16_reduced --data-shape 300
