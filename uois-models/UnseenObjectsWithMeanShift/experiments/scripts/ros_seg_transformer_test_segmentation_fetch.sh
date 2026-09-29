#!/bin/bash
# Usage (from uois-models/UnseenObjectsWithMeanShift, in the msm env):
#   ./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh <gpu_id> <task_name> [--save]
# Subscribes to the Fetch RGB-D topics and publishes MSMFormer predictions on /seg_image,
# /seg_label, /seg_image_refined, ... With --save, frames + predictions go to output/<task_name>/.
#
# Which model is served (default: the pretrained MSMFormer):
#   MODEL      checkpoint (.pth)                   default: data/checkpoints/rgbd_pretrain/norm_RGBD_pretrained.pth
#   MODEL_CFG  its network config (.yaml)          default: MSMFormer/configs/mixture_UCN.yaml
# To serve a model you fine-tuned with MSMFormer/iteach_train_net_pretrained.py --out_dir <out_dir>
# (the same checkpoint/config pairing that lib/fcn/iteach_test_dataset.py evaluates):
#   MODEL=MSMFormer/<out_dir>/model_final.pth MODEL_CFG=MSMFormer/<out_dir>/config.yaml \
#     ./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh 0 <task_name>
# The commented f1/f2 blocks below are the lab's original per-round entries (new_ckpts/f*/).
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

outdir="data/checkpoints"
MODEL=${MODEL:-data/checkpoints/rgbd_pretrain/norm_RGBD_pretrained.pth}
MODEL_CFG=${MODEL_CFG:-MSMFormer/configs/mixture_UCN.yaml}


# f0 (default: pretrained; override with MODEL / MODEL_CFG, see above)
./ros/test_images_segmentation_transformer.py --gpu $1 --task_name $2 $3 \
--cfg experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop.yml \
--pretrained "$MODEL" \
--pretrained_crop data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth \
--network_cfg "$MODEL_CFG" \
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
