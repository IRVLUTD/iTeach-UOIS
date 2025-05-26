#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

# --pretrained MSMFormer/test_sss_rgbd/model_final.pth \
# --pretrained data/checkpoints/rgbd_finetuned/norm_RGBD_finetuned_data04_OCID_5epoch.pth \
# --imgdir data/demo   \
# --color *-color.png   \
# --depth *-depth.png \

./tools/test_image_with_ms_transformer.py  \
--imgdir /home/jishnu/Projects/iTeach-UOIS/uois-models/UnseenObjectsWithMeanShift/data/humanplay_data/test_set/scene47   \
--color rgb/*.png   \
--depth depth/*.png \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
--pretrained MSMFormer/$1/model_0001999.pth \
--pretrained_crop data/checkpoints/rgbd_finetuned/crop_RGBD_finetuned_data04_OSD_1epoch.pth \
--network_cfg MSMFormer/$1/config.yaml \
--network_crop_cfg MSMFormer/configs/crop_mixture_UCN.yaml \
--input_image RGBD_ADD
