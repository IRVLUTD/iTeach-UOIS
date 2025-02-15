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
