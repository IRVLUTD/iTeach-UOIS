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

# Create symlinks for checkpoints
ln -s "$ROOT_DIR/ckpts/checkpoints" "$UCN_DATA_DIR/checkpoints"
ln -s "$ROOT_DIR/ckpts/checkpoints" "$MSM_DATA_DIR/checkpoints"

# Create symlinks for individual checkpoint models
for name in rgb_pretrain rgb_finetuned rgbd_pretrain rgbd_finetuned; do
    ln -s "$ROOT_DIR/ckpts/$name" "$MSM_DATA_DIR/checkpoints/$name"
done

# Tabletop dataset
export TOD_DATA="$DATA_DIR/tabletop_dataset_v5_public"
for dir in "$UCN_DATA_DIR" "$MSM_DATA_DIR"; do
    ln -s "$TOD_DATA" "$dir/tabletop"
done

# OCID dataset
export OCID_DATASET="$DATA_DIR/OCID-dataset"
for dir in "$UCN_DATA_DIR" "$MSM_DATA_DIR"; do
    ln -s "$OCID_DATASET" "$dir/OCID"
done

# TODO: set osd dataset
# export $OCID_dataset=$DATA_DIR/OCID-dataset/
# ln -s $OSD_dataset OSD

# Self-Supervised Segmentation Real-world Dataset
# Ref: (https://irvlutd.github.io/SelfSupervisedSegmentation/)
export SSS_DATA="$DATA_DIR/self-supervised-segmentation"
mv "$SSS_DATA/training" "$SSS_DATA/training_set"
mv "$SSS_DATA/testing" "$SSS_DATA/test_set"

for dir in "$UCN_DATA_DIR" "$MSM_DATA_DIR"; do
    ln -s "$SSS_DATA" "$dir/pushing_data"
done

# TODO: iTeach-UOIS Real-world dataset
export iTEACH_UOIS_DATA="$DATA_DIR/iTeach-UOIS"
