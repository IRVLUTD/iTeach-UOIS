container_name="iteach-uois"
docker rm -f $container_name
DIR=$(pwd)/../
xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name $container_name  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE irvlutd/iteach:uois bash
