#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

# --pretrained MSMFormer/test_sss_rgb_mix/model_0002999.pth \
# --pretrained data/checkpoints/rgb_pretrain/norm_RGB_pretrained.pth \

./tools/test_image_with_ms_transformer.py  \
--imgdir data/demo   \
--color *-color.png   \
--depth *-depth.png \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_color_tabletop.yml \
--pretrained MSMFormer/$1/model_0000499.pth \
--pretrained_crop data/checkpoints/rgb_finetuned/crop_RGB_finetuned_all_5epoch.pth \
--network_cfg MSMFormer/$1/config.yaml \
--network_crop_cfg MSMFormer/configs/crop_mixture_ResNet50.yaml  \
--input_image COLOR
