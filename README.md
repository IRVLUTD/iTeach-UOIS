<div align="center">

# 🧠 iTeach-UOIS

### Unseen Object Instance Segmentation for [iTeach](https://irvlutd.github.io/iTeach/): from gaze prompts to a better robot

<br>

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?style=for-the-badge)](https://irvlutd.github.io/iTeach/)
[![arXiv](https://img.shields.io/badge/arXiv-2410.09072-b31b1b?style=for-the-badge)](https://arxiv.org/abs/2410.09072)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-irvlutd%2Fiteach-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/irvlutd/iteach)

![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![detectron2](https://img.shields.io/badge/detectron2-MSMFormer-6f42c1)
![SAM2](https://img.shields.io/badge/SAM2-video%20propagation-8A2BE2)
![ROS](https://img.shields.io/badge/ROS-Noetic-22314E?logo=ros&logoColor=white)

<br>

[**Data**](#-datasets) &nbsp;·&nbsp;
[**Checkpoints**](#-checkpoints) &nbsp;·&nbsp;
[**Setup**](#️-setup) &nbsp;·&nbsp;
[**GT Masks**](#-generating-ground-truth-masks-for-new-humanplay-scenes) &nbsp;·&nbsp;
[**Training**](#️-msmformer-training) &nbsp;·&nbsp;
[**Live Node**](#-live-ros-node-on-the-robot) &nbsp;·&nbsp;
[**Evaluation**](#-evaluation) &nbsp;·&nbsp;
[**Troubleshooting**](#-known-error-fixes)

</div>

<br>

This repo turns HumanPlay clips labelled on the HoloLens into **dense ground-truth masks** with SAM2, fine-tunes **MSMFormer** on them, and evaluates the result. It also runs MSMFormer as a **live ROS node**, which is how the robot's predictions reach the HoloLens.

```mermaid
flowchart LR
    A["🥽 HumanPlay clip<br/>+ prompts.json"] --> B["🔄 Reverse frames<br/>to JPG"]
    B --> C["🎭 SAM2 propagates<br/>masks backwards"]
    C --> D["🗂️ gt_masks/"]
    D --> E["🏋️ Fine-tune<br/>MSMFormer"]
    E --> F["📊 Evaluate"]
    E --> G["🤖 Deploy as<br/>live ROS node"]
    G -. "next failure" .-> A
```

<br>

## 🧩 Part of the iTeach Family

<table>
  <tr>
    <th width="33%"><a href="https://github.com/IRVLUTD/iTeach">📍 iTeach</a></th>
    <th width="33%"><a href="https://github.com/IRVLUTD/iTeachSkillsApp">🥽 iTeachSkillsApp</a></th>
    <th width="33%">🧠 iTeach-UOIS <sub>(this repo)</sub></th>
  </tr>
  <tr>
    <td>Project hub: overview, links, HoloLens and robot networking utilities</td>
    <td>HoloLens 2 app + ROS bridge: record HumanPlay, gaze + voice point prompts on the last frame</td>
    <td>Point prompts → SAM2 boxes → masks propagated backwards → MSMFormer fine-tuning → evaluation</td>
  </tr>
</table>

<br>

## 📑 Contents

<p align="center">
<a href="#-datasets"><img src="media/toc/01.svg" width="49%" alt="01 · Datasets: Download the data and see where it goes"></a>
<a href="#-checkpoints"><img src="media/toc/02.svg" width="49%" alt="02 · Checkpoints: Pretrained and iTeach fine-tuned weights"></a>
<a href="#️-setup"><img src="media/toc/03.svg" width="49%" alt="03 · Setup: Install with Docker or locally"></a>
<a href="#️-iteach-humanplay-data-layout"><img src="media/toc/04.svg" width="49%" alt="04 · Data Layout: What every scene folder must contain"></a>
<a href="#-generating-ground-truth-masks-for-new-humanplay-scenes"><img src="media/toc/05.svg" width="49%" alt="05 · Ground-Truth Masks: Turn a new capture into labels with SAM2"></a>
<a href="#️-msmformer-training"><img src="media/toc/06.svg" width="49%" alt="06 · Training: Fine-tune MSMFormer: RGB, RGB-D, LoRA"></a>
<a href="#-live-ros-node-on-the-robot"><img src="media/toc/07.svg" width="49%" alt="07 · Live ROS Node: Serve predictions to the HoloLens"></a>
<a href="#-grasping--pick-and-place-scenereplica"><img src="media/toc/08.svg" width="49%" alt="08 · Grasping · Pick &amp; Place: SceneReplica with Contact-GraspNet and GTO"></a>
<a href="#-evaluation"><img src="media/toc/09.svg" width="49%" alt="09 · Evaluation: Score a model on the HumanPlay test set"></a>
<a href="#-known-error-fixes"><img src="media/toc/10.svg" width="49%" alt="10 · Troubleshooting: Fixes for common install and runtime errors"></a>
<a href="#-built-on"><img src="media/toc/more.svg" width="49%" alt="✦ · Credits · License · Cite: Built on, license, citation, contact, thanks"></a>
</p>

<details>
<summary><b>🗂️ Full index</b> <sub>(every section and subsection as text links)</sub></summary>
<br>

<ol>
  <li><a href="#-datasets"><b>Datasets</b></a> · Download the data and see where it goes</li>
  <li><a href="#-checkpoints"><b>Checkpoints</b></a> · Pretrained and iTeach fine-tuned weights</li>
  <li><a href="#️-setup"><b>Setup</b></a> · Install with Docker or locally
    <ul>
    <li><a href="#-option-a-docker-recommended">Docker</a></li>
    <li><a href="#-option-b-local-install">Local install</a></li>
    </ul>
  </li>
  <li><a href="#️-iteach-humanplay-data-layout"><b>Data Layout</b></a> · What every scene folder must contain</li>
  <li><a href="#-generating-ground-truth-masks-for-new-humanplay-scenes"><b>Ground-Truth Masks</b></a> · Turn a new capture into labels with SAM2</li>
  <li><a href="#️-msmformer-training"><b>Training</b></a> · Fine-tune MSMFormer: RGB, RGB-D, LoRA</li>
  <li><a href="#-live-ros-node-on-the-robot"><b>Live ROS Node</b></a> · Serve predictions to the HoloLens</li>
  <li><a href="#-grasping--pick-and-place-scenereplica"><b>Grasping · Pick & Place</b></a> · SceneReplica with Contact-GraspNet and GTO</li>
  <li><a href="#-evaluation"><b>Evaluation</b></a> · Score a model on the HumanPlay test set</li>
  <li><a href="#-known-error-fixes"><b>Troubleshooting</b></a> · Fixes for common install and runtime errors</li>
  <li><a href="#-built-on">Built On</a> · <a href="#-license">License</a> · <a href="#-citation">Citation</a> · <a href="#-contact">Contact</a> · <a href="#-acknowledgements">Thanks</a></li>
</ol>

</details>

<br>

---

<br>

## 📦 Datasets

| Dataset | Scenes | `gt_masks/` | Link |
|:--|:-:|:--|:-:|
| **All UOIS datasets** (TOD, OCID, OSD, RobotPushing, iTeach-HumanPlay) | – | – | [Download](https://utdallas.box.com/v/uois-datasets) |
| iTeach-HumanPlay **D5** · controlled scenes | 5 | ⚠️ recreate (below) | [Download](https://utdallas.box.com/v/iTeach-HumanPlay-D5) |
| iTeach-HumanPlay **D40** | 40 | ✅ included | [Download](https://utdallas.box.com/v/iTeach-HumanPlay-D40) |
| iTeach-HumanPlay **Test** · 902 samples, held-out | 3 | ✅ included | [Download](https://utdallas.box.com/v/iTeach-HumanPlay-Test) |

<br>

Put everything under `DATA/` at the repo root:

```
DATA/
├── tabletop_dataset_v5_public/
├── OCID-dataset/
├── OSD-0.2-depth/                 # copy of OSD-0.2/ (the loaders read the depth variant)
├── self-supervised-segmentation/  # RobotPushing; set_env.sh renames training/ → training_set/, testing/ → test_set/
└── iTeach-HumanPlay/
    ├── training_set/scene*/       # the D5 / D40 scenes you train on
    └── test_set/scene*/           # held-out test scenes
```

> [!IMPORTANT]
> The HumanPlay loader reads **`training_set/scene*/`** and **`test_set/scene*/`**, and each scene needs **`rgb/`**, **`depth/`** and **`gt_masks/`** with identical file names.
> The loader finds each mask and depth image by replacing `rgb` in the image's **full path**. Keep the substring `rgb` out of every other folder name on that path (e.g. not `/home/me/rgbd_work/…`).

> [!WARNING]
> **D5 has no `gt_masks/` folder in the zip.** Its masks are in each scene's `gsam2/masks/` (same file names as `rgb/`); `gt_masks` was a symlink the zip didn't keep. The setup below recreates it in one line.

**Set it up:**

```bash
cd DATA
unzip test_set.zip      -d iTeach-HumanPlay     # → iTeach-HumanPlay/test_set/scene47…49
unzip humanplay-d40.zip -d iTeach-HumanPlay     # → iTeach-HumanPlay/humanplay-d40/scene1…40
unzip humanplay-d5.zip  -d iTeach-HumanPlay     # → iTeach-HumanPlay/humanplay-d5/scene_0423T…

# D5 only: recreate the gt_masks links the zip does not contain
for s in iTeach-HumanPlay/humanplay-d5/scene_*; do ln -sfn gsam2/masks "$s/gt_masks"; done

# Choose the training split (switch between D5 and D40 by re-linking)
ln -sfn humanplay-d40 iTeach-HumanPlay/training_set     # or: ln -sfn humanplay-d5 …
```

**Check it** (after [Setup](#️-setup), in the MSMFormer env). This loads the training split and reads the first sample's `rgb`, `gt_masks` and `depth`:

```bash
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/lib
python test_data.py      # prints the number of training images and a depth shape such as (480, 640)
```

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🔑 Checkpoints

| Checkpoints | Link |
|:--|:--|
| [UCN](https://arxiv.org/pdf/2007.15157) | [Download](https://utdallas.box.com/s/9vt68miar920hf36egeybfflzvt8c676) |
| [MSMFormer](https://arxiv.org/abs/2211.11679) and [Lu et al.](https://roboticsproceedings.org/rss19/p017.pdf) | [Download](https://utdallas.box.com/s/vzp8nmalowg4i58y8b9sghv5s7f36xpz) |
| ✨ **iTeach fine-tuned MSMFormer · D5** | [Download](https://utdallas.box.com/v/iTeach-UOIS-D5-ckpts) |
| ✨ **iTeach fine-tuned MSMFormer · D40** | [Download](https://utdallas.box.com/v/iTeach-UOIS-D40-ckpts) |

<br>

Put them under `ckpts/`. `set_env.sh` expects:

```
ckpts/
├── checkpoints/
├── rgb_pretrain/     rgb_finetuned/
└── rgbd_pretrain/    rgbd_finetuned/
```

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## ⚙️ Setup

```bash
# Clone (the UOIS models are vendored under uois-models/, no submodules needed)
git clone https://github.com/IRVLUTD/iTeach-UOIS
cd iTeach-UOIS

# Environment variables + data/checkpoint symlinks.
# Run from the repo root, after DATA/ and ckpts/ are in place. Safe to re-run.
source ./set_env.sh
```

<sub>Among other things, `set_env.sh` links `DATA/iTeach-HumanPlay` → `uois-models/UnseenObjectsWithMeanShift/data/humanplay_data`, which is where the HumanPlay loader looks.</sub>

<br>

### 🐳 Option A: Docker (recommended)

> [!TIP]
> The Docker image is the lab's reference environment for the UOIS models.

```bash
cd docker
./run_container.sh
```

| | |
|:--|:--|
| **Image** | [`irvlutd/iteach:uois-peft`](https://hub.docker.com/r/irvlutd/iteach) (~32 GB) |
| **Needs** | NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| **Conda envs** | `msm38` and `ucn38` (Python 3.8) for MSMFormer and UCN. The image's last layer is labelled `msm39-peft-lora`; run `conda env list` inside the container to see what is there |

<sub>**No Dockerfile exists for this image.** Its final layers were created with `docker commit` on top of a BundleSDF-based image (layer notes: *"iteach-uois docker image with msmformer and ucn setup"*, then *"msm39-peft-lora"*). The published image itself is the reference.</sub>

<br>

### 🧰 Option B: Local install

You need two environments:

**1. MSMFormer** (lab-tested: conda env `msm39`, Python 3.9, torch 1.10 + CUDA 11.1, detectron2 0.6)
```bash
conda create -n msm39 python=3.9.21 && conda activate msm39
pip install -r uois-models/UnseenObjectsWithMeanShift/requirements-lab-msm39.txt   # exact versions of the lab env
cd uois-models/UnseenObjectsWithMeanShift/MSMFormer/meanshiftformer/modeling/pixel_decoder/ops && sh make.sh   # MSDeformAttn op
```
- For the [live ROS node](#-live-ros-node-on-the-robot), also install system ROS Noetic and `source /opt/ros/noetic/setup.bash`.
- `pip install peft` if you want `--use_lora`
- More details: [MSMFormer README](uois-models/UnseenObjectsWithMeanShift/README.md)

**2. robokit** (SAM2 mask propagation)
- See [robokit/README.md](robokit/README.md)

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🗂️ iTeach-HumanPlay Data Layout

Every scene, whether released or newly captured, looks like this:

```
scene_XXX/
├── rgb/000000.png, 000001.png, …   # 640×480 RGB, zero-padded sequential names in time order
├── depth/000000.png, …             # 16-bit depth in millimetres, same names as rgb/
├── gt_masks/000000.png, …          # instance label maps (0 = background), same names as rgb/
└── prompts.json                    # HoloLens point prompts + SAM2 boxes for the last frame
```

> [!IMPORTANT]
> The loader (`lib/datasets/humanplay_dataset.py`) reads every `rgb/*.png` and finds its label and depth images by replacing `rgb` with `gt_masks` / `depth` in the path. **File names in the three folders must match exactly.**

<br>

<p align="center">
  <img src="media/iteach-data-capture-and-annotation.png" alt="iTeach-UOIS data capture and GT mask generation" width="90%">
  <br>
  <sub><i>iTeach-UOIS data capture and GT mask generation</i></sub>
</p>

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🎭 Generating ground-truth masks for new HumanPlay scenes

> [!NOTE]
> The released D40 and Test sets include `gt_masks/`, and D5 includes the masks in `gsam2/masks/` (see [Datasets](#-datasets) for the one-line link). You only need this section for scenes you capture yourself.

<p align="center">
  <img src="media/sam2-mask-prop.webp" width="85%" alt="SAM2 label propagation">
  <br>
  <sub><i>SAM2 (video mode) propagates masks backwards from the final annotated frame to all earlier frames, producing dense per-frame supervision.</i></sub>
</p>

<br>

**Step 1: Capture and label**

Record and label a scene with [iTeachSkillsApp](https://github.com/IRVLUTD/iTeachSkillsApp). You get `rgb/`, `depth/` and a `prompts.json` with `bboxes_xyxy` (SAM2 boxes on the **last** frame). Arrange them in the [layout above](#️-iteach-humanplay-data-layout).

<br>

**Step 2: Reverse the frames into JPGs**

The last RGB frame becomes `jpg/000000.jpg`, so the prompts on the last frame seed the propagation.

```bash
cd robokit
python convert2jpg_in_reverse.py --input_dir <scene_dir>
```

<br>

**Step 3: Propagate the masks backwards**

```bash
python propogate_masks_via_bbox_prompt_samv2.py --input_dir <scene_dir>   # reads <scene_dir>/jpg/ (the script adds jpg/ itself)
```

This reads `<scene_dir>/prompts.json["bboxes_xyxy"]` and writes `<scene_dir>/gsam2/{masks, palette, bbox_overlay, rgb_and_mask}/`. It then symlinks `<scene_dir>/gt_masks` → `gsam2/masks`, with the masks named in the original (forward) frame order.

<br>

**Step 4: Check the masks**

Look through `gsam2/rgb_and_mask/` before training. 👀

<br>

<sub>`data-preprocessing/get_gsam2_human_labelled_gt_masks.sh` is a record of an earlier sanity-check experiment (GroundingDINO + SAM2). It was <b>not</b> used for any reported result.</sub>

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🏋️ MSMFormer Training

**1. Select the input modality.** Set `__C.INPUT` in `uois-models/UnseenObjectsWithMeanShift/lib/fcn/config.py`:

| Modality | `__C.INPUT` | Config |
|:--|:--|:--|
| RGB | `'COLOR'` | `MSMFormer/configs/humanplay_RGB.yaml` |
| RGB-D | `'RGBD'` | `MSMFormer/configs/humanplay_RGBD.yaml` |

<br>

**2. Train.**

> [!IMPORTANT]
> `humanplay_RGBD.yaml` trains on **`mixture_object_train`**, which mixes the **Tabletop Object Dataset (TOD)** with iTeach-HumanPlay. So RGB-D fine-tuning needs **TOD** in `DATA/tabletop_dataset_v5_public/` (34 GB, from the [UOIS datasets](#-datasets) link) as well as HumanPlay.
>
> **GPU memory:** at `IMS_PER_BATCH: 1`, RGB-D fine-tuning ran out of memory on a 16 GB GPU (RTX 4090 Laptop; about 13.5 GB allocated when it failed). Use a GPU with more memory. Serving the live node plus SAM2 labelling fits in 16 GB (peak ≈ 13.7 GB measured).
>
> **PyTorch 1.10 + setuptools:** training imports PyTorch's TensorBoard writer, which fails with `module 'distutils' has no attribute 'version'` on newer setuptools. Install `setuptools==59.5.0` in the training env (see [Known error fixes](#-known-error-fixes)).

```bash
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/MSMFormer

# RGB fine-tuning on HumanPlay
python tabletop_train_net_pretrained.py --num-gpus 2 --dist-url tcp://127.0.0.1:12345 \
    --cfg $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/MSMFormer/configs/humanplay_RGB.yaml \
    --out_dir test_experiment

# RGB-D fine-tuning on HumanPlay
python iteach_train_net_pretrained.py --num-gpus 1 --dist-url tcp://127.0.0.1:12345 \
    --cfg $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/MSMFormer/configs/humanplay_RGBD.yaml \
    --out_dir test_experiment
```

Checkpoints and `config.yaml` are saved to `uois-models/UnseenObjectsWithMeanShift/MSMFormer/<out_dir>/`. To serve the trained model on the robot, see [Live ROS node](#-live-ros-node-on-the-robot); to score it, see [Evaluation](#-evaluation).

> [!TIP]
> **LoRA:** add `--use_lora` to `iteach_train_net_pretrained.py` to fine-tune with LoRA adapters (needs `pip install peft`). In the original experiments, RGB-D + LoRA failed with a package error.

<br>

**3. See it work.** Qualitative demo on a test scene (run from `uois-models/UnseenObjectsWithMeanShift`):

```bash
./experiments/scripts/hp.iteach.demo_msmformer_rgbd_finetuned.sh <out_dir> data/humanplay_data/test_set/<scene>
./experiments/scripts/hp.iteach.demo_msmformer_rgb_finetuned.sh  <out_dir> data/humanplay_data/test_set/<scene>
```

<sub>These scripts load a fixed iteration (`model_0001999.pth` for RGB-D, `model_0000999.pth` for RGB). Edit `--pretrained` if your run saved different iterations.</sub>

<br>

<p align="center">
  <img src="media/iteach-uois-qual.webp" alt="TableTop and BeyondTableTop scenes" width="90%">
  <br>
  <sub><i>Left → right: ground truth, pretrained MSMFormer, and iTeach fine-tuning rounds FT1, FT3, FT5, on tabletop, shelf and sofa scenes.</i></sub>
</p>

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🤖 Live ROS node on the robot

During an iTeach session, MSMFormer runs on the laptop as a **ROS node**. It subscribes to the Fetch RGB-D topics and publishes predictions (`/seg_image_refined`, `/seg_image`, `/seg_label`, …), which the HoloLens displays via [iTeachSkillsApp](https://github.com/IRVLUTD/iTeachSkillsApp#-running-the-live-system-on-the-robot).

> [!IMPORTANT]
> **Environment:** besides the MSMFormer dependencies, the node imports ROS Python packages: `rospy`, `tf`, `message_filters` and `ros_numpy`. In the lab it runs in the `msm39` env ([Local install](#-option-b-local-install)) with system ROS Noetic sourced; `ros_numpy` comes from the requirements file. The Docker image's contents don't record whether ROS is installed, so check with `python -c "import rospy, tf, message_filters, ros_numpy"` first.

```bash
# The laptop is a ROS client of the robot:
#   ROS_MASTER_URI=http://<robot-ip>:11311   ROS_HOSTNAME=<laptop-ip>
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift

# Pretrained MSMFormer (default)
./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh <gpu_id> <task_name> [--save]

# A model you fine-tuned with --out_dir <out_dir> (see Training)
MODEL=MSMFormer/<out_dir>/model_final.pth MODEL_CFG=MSMFormer/<out_dir>/config.yaml \
  ./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh <gpu_id> <task_name> [--save]
```

| Setting | Effect |
|:--|:--|
| `--save` | Also write frames and predictions to `output/<task_name>/` |
| `MODEL`, `MODEL_CFG` | Checkpoint and its network config. Default: `data/checkpoints/rgbd_pretrain/norm_RGBD_pretrained.pth` + `MSMFormer/configs/mixture_UCN.yaml`. For a fine-tuned run, use its `model_final.pth` + `config.yaml`, the same pairing `lib/fcn/iteach_test_dataset.py` evaluates |
| `f1` / `f2` blocks (commented out) | The lab's original per-round entries (`new_ckpts/f*/model_final.pth`), kept for reference |

<br>

<p align="center">
  <img src="media/realworld-with-gto.webp" width="100%" alt="Real-world pick-and-place with GTO">
  <br>
  <sub><i>Downstream on the robot: iTeach-UOIS segmentation feeds GTO motion planning and grasping, turning perception gains into reliable picks and places.</i></sub>
</p>

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🦾 Grasping & Pick-and-Place (SceneReplica)

The manipulation results use the model-free grasping pipeline from [**SceneReplica**](https://github.com/IRVLUTD/SceneReplica) ([project page](https://irvlutd.github.io/SceneReplica/)). Only the segmentation stage is swapped: iTeach-UOIS's fine-tuned MSMFormer replaces the pretrained one, and everything downstream stays the same.

```mermaid
flowchart LR
    A["🤖 Fetch RGB-D"] --> B["🧠 iTeach-UOIS<br/>MSMFormer node"]
    B -- "segmentation" --> C["✋ Contact-GraspNet<br/>6-DoF grasps"]
    C -- "grasp poses" --> D["🛤️ GTO<br/>trajectory optimisation"]
    D --> E["🦾 Grasp / pick & place"]
```

| # | Component | Repo |
|:-:|:--|:--|
| **1** | 🧠 Segmentation: iTeach-UOIS MSMFormer node | this repo ([Live ROS node](#-live-ros-node-on-the-robot)) |
| **2** | ✋ Grasp proposals: Contact-GraspNet | [IRVLUTD/contact_graspnet](https://github.com/IRVLUTD/contact_graspnet), **`ros` branch**, `contact_graspnet` conda env |
| **3** | 🛤️ Motion planning: GTO | [IRVLUTD/GraspTrajOpt](https://github.com/IRVLUTD/GraspTrajOpt): [running with real robots](https://github.com/IRVLUTD/GraspTrajOpt?tab=readme-ov-file#running-with-real-robots) |
| **4** | 🦾 Benchmark driver | [IRVLUTD/SceneReplica](https://github.com/IRVLUTD/SceneReplica) |

```bash
# 1 · segmentation (this repo; set MODEL / MODEL_CFG for a fine-tuned run)
./experiments/scripts/ros_seg_transformer_test_segmentation_fetch.sh <gpu_id> <task_name>

# 2 · grasp proposals (contact_graspnet, ros branch)
./run_ros_fetch_experiment.sh

# 3 · motion planning: follow GraspTrajOpt's "running with real robots"

# 4 · benchmark driver (SceneReplica)
cd src && python bench_6dof_segmentation_grasping.py \
  --seg_method msmformer --grasp_method contact_gnet --obj_order nearest_first --scene_idx <id>
```

<sub>Set-up of each component (environments, scenes, robot bring-up) follows its own README; see SceneReplica's <a href="https://github.com/IRVLUTD/SceneReplica#model-free-grasping">Model Free Grasping</a> section.</sub>

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 📊 Evaluation

Evaluate a fine-tuned model (`MSMFormer/<out_dir>/model_final.pth`) on the HumanPlay test set (`data/humanplay_data/test_set`):

```bash
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/lib/fcn
python iteach_test_dataset.py <out_dir>
# → MSMFormer/<out_dir>/model_results/results.json
```

To evaluate **every** `model_*.pth` in a set of run directories, use `iteach_get_results_all_models.py`. Adjust the `glob` pattern at the top of its `__main__` block to point at your runs.

<br>

The **combined score** (`lib/fcn/combined_score.py`):

```math
\text{combined} = 0.4 \cdot F_{\text{objects}} + 0.4 \cdot F_{\text{boundary}} + 0.2 \cdot \text{Det}_{@0.75}
```

<sub>where Det@0.75 is the fraction of ground-truth objects detected at 0.75 overlap. All three metrics are fractions in [0, 1]; ×100 gives the percentage scale used by the lab's table script `lib/fcn/j2trex.py`.</sub>

```bash
cd $ROOT_DIR/uois-models/UnseenObjectsWithMeanShift/lib/fcn
python combined_score.py ../../MSMFormer/<out_dir>/model_results/results.json [more results.json …]
# ITEACH-UOIS:
#   1. <out_dir> – 0.xxxx (xx.x)
```

> [!NOTE]
> **Reference result (paper):** iTeach fine-tuning lifts the UOIS combined score from **26.1 → 80.7** (**+54.6**, **3.1×**). Downstream on SceneReplica, grasping success goes **71 → 74** and pick & place **65 → 72** (out of 100).

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🐛 Known Error Fixes

| Error | Fix |
|:--|:--|
| `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'` | `pip install Pillow~=9.5` |
| `AttributeError: module 'distutils' has no attribute 'version'` | `pip install setuptools==59.5.0` |
| `RuntimeError: Numpy is not available` | `pip uninstall numpy && pip install numpy==1.23.1` |

<br>

<div align="right"><sub><a href="#-contents">⬆ back to contents</a></sub></div>

---

<br>

## 🙌 Built On

- [MSMFormer](https://github.com/IRVLUTD/UnseenObjectsWithMeanShift?tab=readme-ov-file#unseen-object-instance-segmentation-with-msmformer)
- [Self-Supervised-UOIS](https://github.com/IRVLUTD/UnseenObjectsWithMeanShift?tab=readme-ov-file#self-supervised-unseen-object-instance-segmentation-via-long-term-robot-interaction)
- [Robokit](https://github.com/jishnujayakumar/robokit)
- [SAM2](https://github.com/facebookresearch/sam2)
- [SceneReplica](https://github.com/IRVLUTD/SceneReplica): grasping and pick-and-place benchmark, with [Contact-GraspNet](https://github.com/IRVLUTD/contact_graspnet) and [GTO](https://github.com/IRVLUTD/GraspTrajOpt)

<sub>[UCN](https://github.com/NVlabs/UnseenObjectClustering) is included under `uois-models/` but was not used in this work. It is kept for possible baseline comparisons or extensions.</sub>

<br>

## 📜 License

Released under the [**MIT License**](LICENSE), © 2024-2026 Intelligent Robotics and Vision Lab (IRVL), The University of Texas at Dallas.

<sub>Third-party code keeps its own license: [MSMFormer](uois-models/UnseenObjectsWithMeanShift/LICENSE.md) (MIT), [UCN](uois-models/UnseenObjectClustering/LICENSE.md) (**NVIDIA Source Code License, non-commercial**), and [robokit](robokit/LICENSE) (MIT).</sub>

<br>

## 📚 Citation

If ***iTeach*** helps your research, please cite:

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

<br>

## 📬 Contact

| | |
|:--|:--|
| 💬 Questions & ideas | [Discussion forum](https://github.com/IRVLUTD/iTeach/discussions) |
| 🛠️ Bugs | [Open an issue](https://github.com/IRVLUTD/iTeach/issues) |
| 📧 Direct | [Jishnu](https://jishnujayakumar.github.io/) |

<br>

## 🙏 Acknowledgements

This work was supported by the DARPA Perceptually-enabled Task Guidance (PTG) Program under contract number HR00112220005, the Sony Research Award Program, and the National Science Foundation (NSF) under Grant No. 2346528. We thank [Sai Haneesh Allu](https://saihaneeshallu.github.io/) for assistance with the real-world experiments.

<br>

<div align="center">
<sub>Built at the <a href="https://labs.utdallas.edu/irvl/">Intelligent Robotics and Vision Lab</a>, The University of Texas at Dallas</sub>
</div>
