#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

# for pytorch port
export MASTER_PORT="1230"

# root dir
export ROOT_DIR=$PWD

# uois-models
# Set root directories
export UOIS_MODEL_DIR="$ROOT_DIR/uois-models"
export UCN_DIR="$UOIS_MODEL_DIR/UnseenObjectClustering"
export MSM_DIR="$UOIS_MODEL_DIR/UnseenObjectsWithMeanShift"
export UCN_DATA_DIR="$UCN_DIR/data"
export MSM_DATA_DIR="$MSM_DIR/data"
export DATA_DIR="$ROOT_DIR/DATA"

# All links below use `ln -sfn`, so re-sourcing this script is safe.
link() { ln -sfn "$1" "$2"; }

# Create symlinks for checkpoints
link "$ROOT_DIR/ckpts/checkpoints" "$UCN_DATA_DIR/checkpoints"
link "$ROOT_DIR/ckpts/checkpoints" "$MSM_DATA_DIR/checkpoints"

# Create symlinks for individual checkpoint models
for name in rgb_pretrain rgb_finetuned rgbd_pretrain rgbd_finetuned; do
    link "$ROOT_DIR/ckpts/$name" "$MSM_DATA_DIR/checkpoints/$name"
done

# Tabletop dataset
export TOD_DATA="$DATA_DIR/tabletop_dataset_v5_public"
for dir in "$UCN_DATA_DIR" "$MSM_DATA_DIR"; do
    link "$TOD_DATA" "$dir/tabletop"
done

# OCID dataset
export OCID_DATASET="$DATA_DIR/OCID-dataset"
for dir in "$UCN_DATA_DIR" "$MSM_DATA_DIR"; do
    link "$OCID_DATASET" "$dir/OCID"
done

# OSD dataset (loaders read data/OSD)
export OSD_DATASET="$DATA_DIR/OSD-0.2-depth"
for dir in "$UCN_DATA_DIR" "$MSM_DATA_DIR"; do
    link "$OSD_DATASET" "$dir/OSD"
done

# Self-Supervised Segmentation Real-world Dataset
# Ref: (https://irvlutd.github.io/SelfSupervisedSegmentation/)
export SSS_DATA="$DATA_DIR/self-supervised-segmentation"
# The loaders expect training_set/ and test_set/; rename only once
[ -d "$SSS_DATA/training" ] && [ ! -e "$SSS_DATA/training_set" ] && mv "$SSS_DATA/training" "$SSS_DATA/training_set"
[ -d "$SSS_DATA/testing" ] && [ ! -e "$SSS_DATA/test_set" ] && mv "$SSS_DATA/testing" "$SSS_DATA/test_set"

for dir in "$UCN_DATA_DIR" "$MSM_DATA_DIR"; do
    link "$SSS_DATA" "$dir/pushing_data"
done

# iTeach-HumanPlay dataset (loader reads data/humanplay_data/{training_set,test_set}/scene*/)
export iTEACH_UOIS_DATA="$DATA_DIR/iTeach-HumanPlay"
link "$iTEACH_UOIS_DATA" "$MSM_DATA_DIR/humanplay_data"
