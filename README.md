
# iTeach-UOIS

Unseen Object Instance Segmentation (UOIS) part of [iTeach](https://irvlutd.github.io/iTeach/). It covers SAM2 mask propagation for HumanPlay clips, MSMFormer fine-tuning and evaluation.

iTeach is split across three repositories:

| Repo | Role in the pipeline |
|---|---|
| [IRVLUTD/iTeach](https://github.com/IRVLUTD/iTeach) | Project hub (overview, links, HoloLens/robot networking utilities) |
| [IRVLUTD/iTeachSkillsApp](https://github.com/IRVLUTD/iTeachSkillsApp) | HoloLens 2 app + ROS bridge: record a HumanPlay clip, place point prompts on the last frame with eye-gaze + voice |
| **IRVLUTD/iTeach-UOIS** (this repo) | Point prompts → SAM2 boxes → masks propagated backwards → MSMFormer fine-tuning → evaluation |

## Contents
- [📦 Datasets](#-datasets)
- [🔑 Checkpoints](#-checkpoints)
- [⚙️ Setup](#️-setup)
- [🐳 Docker](#-docker)
- [🗂️ iTeach-HumanPlay data layout](#️-iteach-humanplay-data-layout)
- [🎭 Generating ground-truth masks for new HumanPlay scenes](#-generating-ground-truth-masks-for-new-humanplay-scenes)
- [🏋️‍♂️ MSMFormer training (RGB, RGBD, LoRA)](#️️-msmformer-training-rgb-rgbd-lora)
- [🤖 Live ROS node on the robot](#-live-ros-node-on-the-robot)
- [📊 Evaluation](#-evaluation)
- [🐛 Known error fixes](#-known-error-fixes)

## 📦 Datasets
- Download TOD, OCID, OSD, RobotPushing, iTeach-HumanPlay datasets from [here](https://utdallas.box.com/v/uois-datasets).
- iTeach-HumanPlay can also be downloaded separately:
  - **D5** (5 controlled scenes): [Link](https://utdallas.box.com/v/iTeach-HumanPlay-D5)
  - **D40** (40 scenes): [Link](https://utdallas.box.com/v/iTeach-HumanPlay-D40)
  - **Test set** (902 samples from 3 held-out scenes): [Link](https://utdallas.box.com/v/iTeach-HumanPlay-Test)
- Put all the data in the `DATA/` directory at the repo root, so that it looks like:
  ```
  DATA/
  ├── tabletop_dataset_v5_public/
  ├── OCID-dataset/
  ├── OSD-0.2-depth/                 # copy of OSD-0.2/ (the loaders read the depth variant)
  ├── self-supervised-segmentation/  # RobotPushing; set_env.sh renames training/ → training_set/, testing/ → test_set/
  └── iTeach-HumanPlay/
      ├── training_set/scene*/
      └── test_set/scene*/
  ```
  The HumanPlay folders must be named `training_set/` and `test_set/`, with scene folders named `scene*`. Put the D5/D40 scenes you train on under `training_set/`, and the test scenes under `test_set/`.

## 🔑 Checkpoints
- Download [UCN](https://arxiv.org/pdf/2007.15157) checkpoints from [here](https://utdallas.box.com/s/9vt68miar920hf36egeybfflzvt8c676).
- Download [MSM](https://arxiv.org/abs/2211.11679) and [Lu et al.](https://roboticsproceedings.org/rss19/p017.pdf) checkpoints from [here](https://utdallas.box.com/s/vzp8nmalowg4i58y8b9sghv5s7f36xpz).
- Put all the checkpoints in the `ckpts/` directory. `set_env.sh` expects `ckpts/checkpoints/` plus `ckpts/{rgb_pretrain,rgb_finetuned,rgbd_pretrain,rgbd_finetuned}/`.
- ✨ iTeach fine-tuned MSMFormer checkpoints:
  - **D5**: [Link](https://utdallas.box.com/v/iTeach-UOIS-D5-ckpts)
  - **D40**: [Link](https://utdallas.box.com/v/iTeach-UOIS-D40-ckpts)

## ⚙️ Setup
```bash
# Clone the repo (the UOIS models are vendored under uois-models/, no submodules needed)
git clone https://github.com/IRVLUTD/iTeach-UOIS
cd iTeach-UOIS

# Set environment variables and create the data/checkpoint symlinks.
# Run from the repo root, after DATA/ and ckpts/ are in place. Safe to re-run.
source ./set_env.sh
```

`set_env.sh` links `DATA/iTeach-HumanPlay` to `uois-models/UnseenObjectsWithMeanShift/data/humanplay_data`, which is where the HumanPlay loader looks.

Without Docker, you need two environments:

- **MSMFormer**: Python 3.8, a CUDA build of PyTorch, and [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) matching that PyTorch/CUDA version. Then run `pip install -r uois-models/UnseenObjectsWithMeanShift/requirement.txt` and compile the MSDeformAttn op in `uois-models/UnseenObjectsWithMeanShift/MSMFormer/meanshiftformer/modeling/pixel_decoder/ops` (`sh make.sh`). See the [MSMFormer README](uois-models/UnseenObjectsWithMeanShift/README.md). `--use_lora` also needs `peft`.
- **robokit** (SAM2 mask propagation): see [robokit/README.md](robokit/README.md).

The exact package versions used for the paper are baked into the Docker image below. Using the image is the most reliable way to reproduce the results.

## 🐳 Docker
A [docker image](https://hub.docker.com/r/irvlutd/iteach) is provided. The tag used is `irvlutd/iteach:uois-peft` (~32 GB). It needs an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
```bash
cd docker
./run_container.sh
```
There are two Conda environments for the UOIS models (Python 3.8):
- `msm38` → `/opt/conda/envs/msm38`
- `ucn38` → `/opt/conda/envs/ucn38`

> Note: the Dockerfile for `irvlutd/iteach:uois-peft` is not in this repository. The published image is the reference environment.

## 🗂️ iTeach-HumanPlay data layout
Each scene (released or newly captured) looks like this:
```
scene_XXX/
├── rgb/000000.png, 000001.png, ...   # 640x480 RGB, zero-padded sequential names in time order
├── depth/000000.png, ...             # 16-bit depth in millimetres, same names as rgb/
├── gt_masks/000000.png, ...          # instance label maps (0 = background), same names as rgb/
└── prompts.json                      # HoloLens point prompts + SAM2 boxes for the last frame
```
The loader (`lib/datasets/humanplay_dataset.py`) reads every `rgb/*.png` and finds the label and depth images by replacing `rgb` with `gt_masks` / `depth` in the path. The file names in the three folders must therefore match exactly.

![alt text](media/iteach-data-capture-and-annotation.png)

<div align="center">iTeach-UOIS data capture and GT mask generation</div>

## 🎭 Generating ground-truth masks for new HumanPlay scenes
The released datasets already include `gt_masks/`. Follow this section only when you capture new scenes.

1. **Capture and label** a scene with [iTeachSkillsApp](https://github.com/IRVLUTD/iTeachSkillsApp). This produces `rgb/`, `depth/` and a `prompts.json` that contains `bboxes_xyxy` (SAM2 boxes on the **last** frame). Put them into the scene layout above.
2. **Reverse the frames into JPGs** for SAM2. The last RGB frame becomes `jpg/000000.jpg`, so the prompts on the last frame seed the propagation:
   ```bash
   cd robokit
   python convert2jpg_in_reverse.py --input_dir <scene_dir>
   ```
3. **Propagate the masks backwards** through the clip:
   ```bash
   python propogate_masks_via_bbox_prompt_samv2.py --input_dir <scene_dir>/jpg
   ```
   This reads `<scene_dir>/prompts.json["bboxes_xyxy"]` and writes `<scene_dir>/gsam2/{masks,palette,bbox_overlay,rgb_and_mask}/`. It then symlinks `<scene_dir>/gt_masks` to `gsam2/masks`, with the masks named in the original (forward) frame order.
4. Check `gsam2/rgb_and_mask/` visually before training.

`data-preprocessing/get_gsam2_human_labelled_gt_masks.sh` is a record of an earlier sanity-check experiment (GroundingDINO + SAM2). It was **not** used for any reported result.

## 🏋️‍♂️ MSMFormer training (RGB, RGBD, LoRA)
Set `__C.INPUT` to `'COLOR'` (RGB) or `'RGBD'` in:
```bash
$ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/lib/fcn/config.py
```
It must match the config you train with. `humanplay_RGB.yaml` is the RGB config and `humanplay_RGBD.yaml` the RGBD one.

```bash
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/MSMFormer

# RGB fine-tuning on HumanPlay
python tabletop_train_net_pretrained.py --num-gpus 2 --dist-url tcp://127.0.0.1:12345 \
    --cfg $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/MSMFormer/configs/humanplay_RGB.yaml --out_dir test_experiment

# RGBD fine-tuning on HumanPlay
python iteach_train_net_pretrained.py --num-gpus 1 --dist-url tcp://127.0.0.1:12345 \
    --cfg $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/MSMFormer/configs/humanplay_RGBD.yaml --out_dir test_experiment

# Optional (iteach_train_net_pretrained.py only): add --use_lora to fine-tune with LoRA adapters (needs `pip install peft`).
# Note from the original experiments: RGBD + LoRA failed with a package error.
```
Checkpoints and `config.yaml` are saved to `uois-models/UnseenObjectsWithMeanShift/MSMFormer/<out_dir>/`.

Qualitative demo on a test scene, using a trained `<out_dir>`. Run from `uois-models/UnseenObjectsWithMeanShift`:
```bash
./experiments/scripts/hp.iteach.demo_msmformer_rgbd_finetuned.sh <out_dir> data/humanplay_data/test_set/<scene>
./experiments/scripts/hp.iteach.demo_msmformer_rgb_finetuned.sh  <out_dir> data/humanplay_data/test_set/<scene>
```
These scripts load a fixed iteration (`model_0001999.pth` for RGBD, `model_0000999.pth` for RGB). Edit `--pretrained` if your run saved different iterations.

![alt text](media/iteach-uois-qual.webp)
<div align="center">TableTop and BeyondTableTop: (Shelf, Sofa) scenes</div>

## 🤖 Live ROS node on the robot
During an iTeach session, MSMFormer runs on the laptop as a ROS node. It subscribes to the Fetch RGB-D topics and publishes predictions (`/seg_image_refined`, `/seg_image`, `/seg_label`, …), which the HoloLens displays via [iTeachSkillsApp](https://github.com/IRVLUTD/iTeachSkillsApp#running-the-live-system-on-the-robot):
```bash
# ROS client of the robot: ROS_MASTER_URI=http://<robot-ip>:11311, ROS_HOSTNAME=<laptop-ip>
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift
./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh <gpu_id> <task_name> [--save]
```
`--save` also writes frames and predictions to `output/<task_name>/`. The script's `f0` block uses the pretrained model. Switch to the `f1`/`f2` blocks (`new_ckpts/f*/model_final.pth`) to run the models fine-tuned after each iTeach round.

## 📊 Evaluation
Evaluate a fine-tuned model (`MSMFormer/<out_dir>/model_final.pth`) on the HumanPlay test set (`data/humanplay_data/test_set`):
```bash
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/lib/fcn
python iteach_test_dataset.py <out_dir>
# → MSMFormer/<out_dir>/model_results/results.json
```
`iteach_get_results_all_models.py` does the same for every `model_*.pth` in a set of run directories. Adjust the `glob` pattern at the top of its `__main__` block to point at your runs.

The **combined score** is computed in `lib/fcn/combined_score.py`:

```
combined = 0.4 · Objects F-measure + 0.4 · Boundary F-measure + 0.2 · (% objects detected at 0.75)
```

## 🐛 Known error fixes
If you encounter:
- `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'`, run:
  ```bash
  pip install Pillow~=9.5
  ```
- `AttributeError: module 'distutils' has no attribute 'version'`, run:
  ```bash
  pip install setuptools==59.5.0
  ```
- `RuntimeError: Numpy is not available`, run:
  ```bash
  pip uninstall numpy && pip install numpy==1.23.1
  ```

## 🙌 Works used
- [MSMFormer](https://github.com/IRVLUTD/UnseenObjectsWithMeanShift?tab=readme-ov-file#unseen-object-instance-segmentation-with-msmformer)
- [Self-Supervised-UOIS](https://github.com/IRVLUTD/UnseenObjectsWithMeanShift?tab=readme-ov-file#self-supervised-unseen-object-instance-segmentation-via-long-term-robot-interaction)
- [Robokit](https://github.com/jishnujayakumar/robokit)
- [SAM2](https://github.com/facebookresearch/sam2)
- Note: [UCN](https://github.com/NVlabs/UnseenObjectClustering) is included under `uois-models/` but was not used in this work.
  - It is kept for possible future baseline comparisons or extensions.


## 📚 BibTex
Please cite ***iTeach*** if it helps your research 🙌:
```bibtex
@misc{padalunkal2024iteach,
  title         = {iTeach: In the Wild Interactive Teaching for Failure-Driven Adaptation of Robot Perception},
  author        = {Jishnu Jaykumar P and Cole Salvato and Vinaya Bomnale and Jikai Wang and Ayush Bhardwaj and Jin-Ryong Kim and Yu Xiang},
  year          = {2026},
  eprint        = {2410.09072},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2410.09072}
}
```

## 📬 Contact
For any clarification, comments, or suggestions, you can choose from the following options:

- Join the [discussion forum](https://github.com/IRVLUTD/iTeach/discussions). 💬
- Report an [issue](https://github.com/IRVLUTD/iTeach/issues). 🛠️
- Contact [Jishnu](https://jishnujayakumar.github.io/). 📧

## 🙏 Acknowledgements
This work was supported by the DARPA Perceptually-enabled Task Guidance (PTG) Program under contract number HR00112220005, the Sony Research Award Program, and the National Science Foundation (NSF) under Grant No.2346528. We thank [Sai Haneesh Allu](https://saihaneeshallu.github.io/) for assistance with the real-world experiments. 🙌
