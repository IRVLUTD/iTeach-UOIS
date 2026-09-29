# 🎭 SAM2 Mask Propagation (robokit)

This folder is a copy of [robokit](https://github.com/jishnujayakumar/robokit) (general docs: [README.robokit.md](README.robokit.md)), plus the scripts that turn a labelled HumanPlay clip into **dense ground-truth masks** for iTeach-UOIS.

<br>

## 📦 Install

Requires Python ≥ 3.9 with a CUDA build of PyTorch.

```bash
cd robokit
export CUDA_HOME=/usr/local/cuda

pip install -r requirements.txt
python setup.py install   # also installs GroundingDINO, MobileSAM and SAM2 (pinned commit) and downloads their checkpoints
```

<br>

## 🔄 1 · Prepare the frames for SAM2

Convert the PNG frames to JPGs in **reverse** order, so the last frame (where the HoloLens prompts were placed) becomes `jpg/000000.jpg`:

```bash
python convert2jpg_in_reverse.py --input_dir <scene_dir>   # reads <scene_dir>/rgb/*.png → writes <scene_dir>/jpg/
```

<br>

## 🎭 2 · Propagate the masks backwards

```bash
python propogate_masks_via_bbox_prompt_samv2.py --input_dir <scene_dir>   # reads <scene_dir>/jpg/ (the script adds jpg/ itself)
```

| Reads | Writes |
|:--|:--|
| `<scene_dir>/prompts.json` → `bboxes_xyxy` (pixel `[x1, y1, x2, y2]` on the last frame, from [iTeachSkillsApp](https://github.com/IRVLUTD/iTeachSkillsApp)) | `<scene_dir>/gsam2/` masks and visualizations, plus a `<scene_dir>/gt_masks` symlink |

<br>

## 📊 3 · Test a checkpoint on the dataset

Run this in the MSMFormer environment (`msm38` in the Docker image):

```bash
cd ../uois-models/UnseenObjectsWithMeanShift/lib/fcn
python iteach_test_dataset.py <MSMFormer_out_dir>
```
