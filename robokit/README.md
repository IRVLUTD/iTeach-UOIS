## iTeach-UOIS: SAM2 mask propagation (robokit)

This folder is a copy of [robokit](https://github.com/jishnujayakumar/robokit) (general docs: [README.robokit.md](README.robokit.md)), plus the scripts that turn a labelled HumanPlay clip into dense ground-truth masks.

### Install
Python ≥ 3.9 with a CUDA build of PyTorch, and `CUDA_HOME` set:
```bash
cd robokit
export CUDA_HOME=/usr/local/cuda
pip install -r requirements.txt
python setup.py install   # also installs GroundingDINO, MobileSAM and SAM2 (pinned commit) and downloads their checkpoints
```

### Prepare data for SAM2
Convert the PNG frames to JPGs in reverse order. The last frame, where the HoloLens prompts were placed, becomes `jpg/000000.jpg`:
```bash
python convert2jpg_in_reverse.py --input_dir <scene_dir>        # reads <scene_dir>/rgb/*.png, writes <scene_dir>/jpg/
```

### Propagate masks in reverse
```bash
python propogate_masks_via_bbox_prompt_samv2.py --input_dir <scene_dir>/jpg
```
This reads the boxes from `<scene_dir>/prompts.json` (key `bboxes_xyxy`, pixel `[x1, y1, x2, y2]` on the last frame, written by [iTeachSkillsApp](https://github.com/IRVLUTD/iTeachSkillsApp)). It writes the masks to `<scene_dir>/gsam2/` and links `<scene_dir>/gt_masks` to them.

### Test a model checkpoint on the dataset
Run in the MSMFormer environment (`msm38` in the Docker image):
```bash
cd ../uois-models/UnseenObjectsWithMeanShift/lib/fcn
python iteach_test_dataset.py <MSMFormer_out_dir>
```
