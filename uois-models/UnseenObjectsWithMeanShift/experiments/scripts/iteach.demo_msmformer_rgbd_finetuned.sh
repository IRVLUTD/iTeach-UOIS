#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

# Pretrained MSMFormer

# ./tools/test_image_with_ms_transformer.py  \
# --imgdir data/demo   \
# --color *-color.png   \
# --depth *-depth.png \
# --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
# --pretrained data/checkpoints/rgbd_pretrain/norm_RGBD_pretrained.pth \
# --pretrained_crop data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth \
# --network_cfg data/checkpoints/rgbd_pretrain/mixture_UCN.yaml \
# --network_crop_cfg data/checkpoints/rgbd_pretrain/crop_mixture_UCN.yaml \
# --input_image RGBD_ADD


./tools/test_image_with_ms_transformer.py  \
--imgdir /home/jishnu/Desktop/iros25-submission/iros25-sub-rw/gt-msm-iteach-comparision/first-frame   \
--color color-*.png   \
--depth depth-*.png \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
--pretrained data/checkpoints/rgbd_pretrain/norm_RGBD_pretrained.pth \
--pretrained_crop data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth \
--network_cfg data/checkpoints/rgbd_pretrain/mixture_UCN.yaml \
--network_crop_cfg data/checkpoints/rgbd_pretrain/crop_mixture_UCN.yaml \
--input_image RGBD_ADD


# iTeach-UOIS

# ckpt_dir="MSMFormer/human_play_rgbd_f2_mix_2120_250"


# ./tools/test_image_with_ms_transformer.py  \
# --imgdir /home/jishnu/Desktop/iros25-submission/iros25-sub-rw/gt-msm-iteach-comparision/last-frame   \
# --color color-*.png   \
# --depth depth-*.png \
# --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
# --pretrained $ckpt_dir/model_0000999.pth \
# --pretrained_crop data/checkpoints/rgbd_finetuned/crop_RGBD_finetuned_data04_OSD_1epoch.pth \
# --network_cfg $ckpt_dir/config.yaml \
# --network_crop_cfg MSMFormer/configs/crop_mixture_UCN.yaml \
# --input_image RGBD_ADD
