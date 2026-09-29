#!/bin/bash
# Usage (from uois-models/UnseenObjectsWithMeanShift, in the msm env):
#   ./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh <gpu_id> <task_name> [--save]
# Subscribes to the Fetch RGB-D topics and publishes MSMFormer predictions on /seg_image,
# /seg_label, /seg_image_refined, ... With --save, frames + predictions go to output/<task_name>/.
# f0 = pretrained model; f1/f2 = models fine-tuned after each iTeach round (swap the block below).
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir="data/checkpoints"


# f0
./ros/test_images_segmentation_transformer.py --gpu $1 --task_name $2 $3 \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
--pretrained data/checkpoints/rgbd_pretrain/norm_RGBD_pretrained.pth \
--pretrained_crop data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth \
--network_cfg MSMFormer/configs/mixture_UCN.yaml \
--network_crop_cfg MSMFormer/configs/crop_mixture_UCN.yaml \
--input_image RGBD_ADD \
--camera Fetch \
# --no_refinement     # comment this out if you want to use labe refinement

# f1
# ./ros/test_images_segmentation_transformer.py --gpu $1 --task_name $2 $3 \
# --cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
# --pretrained ./new_ckpts/f1/model_final.pth \
# --pretrained_crop data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth \
# --network_cfg MSMFormer/configs/mixture_UCN.yaml \
# --network_crop_cfg MSMFormer/configs/crop_mixture_UCN.yaml \
# --input_image RGBD_ADD \
# --camera Fetch \
#--no_refinement     # comment this out if you want to use labe refinement

# f2
#./ros/test_images_segmentation_transformer.py --gpu $1 --task_name $2 $3 \
#--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
#--pretrained ./new_ckpts/f2/model_final.pth \
#--pretrained_crop data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth \
#--network_cfg MSMFormer/configs/mixture_UCN.yaml \
#--network_crop_cfg MSMFormer/configs/crop_mixture_UCN.yaml \
#--input_image RGBD_ADD \
#--camera Fetch \
#--no_refinement     # comment this out if you want to use labe refinement
