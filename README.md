# iTeach-UOIS



## Datasets
- Download TOD, OCID, OSD, RealWorld, iTeach-RealWorld datasets from [here](https://utdallas.box.com/v/uois-datasets).
- Put all the data in `DATA/` directory


## Checkpoints
- Download UCN ckpts from [here](https://utdallas.box.com/s/9vt68miar920hf36egeybfflzvt8c676).
- Download MSM and SSS ckpts from [here](https://utdallas.box.com/s/vzp8nmalowg4i58y8b9sghv5s7f36xpz).
- Put all the checkpoints in `ckpts/` directory

## Setup
```shell
# git clone 
git clone --recurse-submodules https://github.com/jishnujayakumar/iTeach-UOIS

# set env vars
source ./set_env.sh
```

## Docker
We provide a [docker image](https://hub.docker.com/repository/docker/irvlutd/iteach):
```shell
cd docker
./run_container.sh;
```
There are two conda envs for respective uois-models with py3.8
- `msm38` -> `/opt/conda/envs/msm38`
- `ucn38` -> `/opt/conda/envs/ucn38`


## Known Error Fixes
If there is an error like:
- *AttributeError: module 'PIL.Image' has no attribute 'LINEAR'*, try: `pip install Pillow~=9.5`
- *AttributeError: module 'distutils' has no attribute 'version'*, try: `pip install setuptools==59.5.0`


```shell
FileNotFoundError: [Errno 2] No such file or directory: 'data/pushing_data/training_set/0102T145631/mat-000000.mat'
>>> data = loadmat("data/pushing_data/training_set/0102T145631/meta-000000.mat")
>>> data
{'__header__': b'MATLAB 5.0 MAT-file Platform: posix, Created on: Mon Jan  2 14:56:31 2023', '__version__': '1.0', '__globals__': [], 'intrinsic_matrix': array([[530.15079807,   0.        , 321.85101905],
       [  0.        , 527.83633424, 232.7456859 ],
       [  0.        ,   0.        ,   1.        ]]), 'factor_depth': array([[1000.]]), 'camera_pose': array([[ 0.00480202, -0.76207565,  0.64747018,  0.15526595],
       [-0.99996539, -0.00805851, -0.00206857,  0.02258708],
       [ 0.00679405, -0.64743784, -0.76208798,  1.40455238],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])}
```


Results

- MSM+Mixture+Learnable-Backbone
```javascript
{'Objects F-measure': 0.03765690376569038, 'Objects Precision': 1.0, 'Objects Recall': 0.03765690376569038, 'Boundary F-measure': 0.03765690376569038, 'Boundary Precision': 1.0, 'Boundary Recall': 0.03765690376569038, 'obj_detected': 0.0, 'obj_detected_075': 0.0, 'obj_gt': 7.4715481171548115, 'obj_detected_075_percentage': 0.03765690376569038}
```
