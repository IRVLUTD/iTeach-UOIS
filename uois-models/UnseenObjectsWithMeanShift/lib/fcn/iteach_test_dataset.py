import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "MSMFormer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "datasets"))
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from meanshiftformer.config import add_meanshiftformer_config
from datasets import OCIDDataset, OSDObject
from datasets.tabletop_dataset import TableTopDataset, getTabletopDataset
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from tabletop_config import add_tabletop_config
from datasets.pushing_dataset import PushingDataset
from datasets.humanplay_dataset import HumanPlayDataset
from datasets.uoais_dataset import UOAIS_Dataset
from datasets.load_OSD_UOAIS import OSDObject_UOAIS

# ignore some warnings
import warnings
import torch
from config import cfg

warnings.simplefilter("ignore", UserWarning)
from test_utils import (
    test_dataset,
    test_sample,
    test_sample_crop,
    test_dataset_crop,
    Network_RGBD,
    test_sample_crop_nolabel,
)

import logging


def get_general_predictor(cfg_file, weight_path, input_image="RGBD_ADD"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    cfg_file = cfg_file
    cfg.USE_LoRA = False
    cfg.merge_from_file(cfg_file)
    add_tabletop_config(cfg)
    cfg.SOLVER.IMS_PER_BATCH = 1  #

    cfg.INPUT.INPUT_IMAGE = input_image
    if input_image == "RGBD_ADD":
        cfg.MODEL.USE_DEPTH = True
    else:
        cfg.MODEL.USE_DEPTH = False
    # arguments frequently tuned
    cfg.TEST.DETECTIONS_PER_IMAGE = 20
    weight_path = weight_path
    cfg.MODEL.WEIGHTS = weight_path
    predictor = Network_RGBD(cfg)
    return predictor, cfg


def get_predictor(
    cfg_file,
    weight_path,
    input_image="RGBD_ADD",
):
    return get_general_predictor(cfg_file, weight_path, input_image=input_image)


def get_predictor_crop(
    cfg_file,
    weight_path,
    input_image="RGBD_ADD",
):
    return get_general_predictor(cfg_file, weight_path, input_image=input_image)


# set datasets

# use_my_dataset = True
# for d in ["train", "test"]:
#     if use_my_dataset:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: TableTopDataset(d))
#     else:
#         DatasetCatalog.register("tabletop_object_" + d, lambda d=d: getTabletopDataset(d))

metadata = MetadataCatalog.get("tabletop_object_train")


if __name__ == "__main__":
    # Here you can set the paths for networks
    dirname = os.path.dirname(__file__)

    # # RGBD
    # cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/human_play_rgb_f2_mix/config.yaml')
    # weight_path_MSMFormer = os.path.join(dirname, "../../MSMFormer/human_play_rgb_f2_mix/model_final.pth")
    # weight_path_MSMFormer = os.path.join(dirname, "../../data/rgbd_pretrain/norm_RGBD_pretrained.pth")


    # weight_path_MSMFormer = os.path.join(dirname, "../../MSMFormer/norm_0111_RGB_mixture2_updated/model_0000319.pth")


    # cfg_file_MSMFormer = os.path.join(dirname, '../../MSMFormer/configs/UOAIS_ResNet50.yaml')
    # weight_path_MSMFormer = os.path.join(dirname, "../../data/checkpoints/OSD_RGB_MSMFormer_UOAIS_SIM.pth")
    #

    #  ======================================= RGBD =================================================

    finetuning_model_dir = sys.argv[1]

    cfg_model = [(
        os.path.join(dirname, "../../MSMFormer/configs/mixture_UCN.yaml"),
        os.path.join(dirname, "../../data/rgbd_pretrain/norm_RGBD_pretrained.pth")
    ),
    (
        os.path.join(dirname, f"../../MSMFormer/{finetuning_model_dir}/config.yaml"),
        os.path.join(dirname, f"../../MSMFormer/{finetuning_model_dir}/model_final.pth")
    )]

    # Pretrained model
    cfg_file_MSMFormer = cfg_model[1][0]
    weight_path_MSMFormer = cfg_model[1][1]

    # Finetuned model
    # cfg_file_MSMFormer = cfg_model[1][0]
    # weight_path_MSMFormer = cfg_model[1][1]

    # Always use pretrained model for 2nd stage crop model
    cfg_file_MSMFormer_crop = os.path.join(
        dirname, "../../MSMFormer/configs/crop_mixture_UCN.yaml"
    )
    weight_path_MSMFormer_crop = os.path.join(
        dirname, "../../data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth"
    )
    #  ======================================= RGBD =================================================
    
    # osd_dataset = OSDObject_UOAIS(image_set="test")
 
 
    # ocid_dataset = OCIDDataset(image_set="test")
    # osd_dataset = OSDObject(image_set="test")


    # print(osd_dataset[0]['depth'])
    # pushing_dataset = PushingDataset("test")
    # dataset = pushing_dataset

    humanplay_dataset = HumanPlayDataset("test")
    dataset = humanplay_dataset

    # uoais_dataset = UOAIS_Dataset("train")
    # predictor, cfg = get_predictor(
    #     cfg_file=cfg_file_MSMFormer,
    #     weight_path=weight_path_MSMFormer,
    #     #    input_image = "COLOR"
    #     input_image="RGBD_ADD",
    # )

    # for getting results for all model results on pushing dataset
    # cd lib/fcn first
    # Finetuned model from 0
    import glob

    directory = f"../../MSMFormer/{finetuning_model_dir}"

    model_results = f"{directory}/model_results"

    os.makedirs(model_results, exist_ok=True)
    
    # Get all .pth files
    pth_files = glob.glob(f"{directory}/model_*")

    # Sorting function that puts 'final' at the end
    def sort_key(filename):
        basename = os.path.basename(filename)
        try:
            # Extract the number after "model_"
            return int(basename.split('_')[1].split('.')[0])
        except ValueError:
            return float('inf')  # Place 'final' at the end

    # Sort the files
    pth_files_sorted = sorted(pth_files, key=sort_key)

    from tqdm import tqdm
    import json

    print("******************************** START ********************************")
    print(cfg_file_MSMFormer)
    print(weight_path_MSMFormer)



    predictor, cfg = get_predictor(cfg_file=cfg_file_MSMFormer,
                                weight_path=weight_path_MSMFormer,
                                )
    
    results = test_dataset(cfg, dataset, predictor)

    # predictor_crop, cfg_crop = get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop,
    #                                             weight_path=weight_path_MSMFormer_crop)

    # results_refined = test_dataset_crop(cfg, dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)

    # pth_result = {
    #     "results": results,
    #     # "results_refined": results_refined
    # }

    # print(pth_result.keys())

    json_path = os.path.join(model_results, "results.json")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # import pdb; pdb.set_trace()

    print("********************************  END  ********************************")
    
    #########################################################################################3

    # test_sample(cfg, uoais_dataset[0], predictor, visualization=True)
    # for i in range(10, 20):
    #     test_sample(cfg, osd_dataset[i], predictor, visualization=True, topk=False, confident_score=0.7)
    # test_dataset(cfg, osd_dataset, predictor)

    # predictor, cfg = get_predictor(cfg_file=cfg_file_MSMFormer,
    #                                weight_path=weight_path_MSMFormer,
    #                                # input_image = "COLOR"
    #                                )
    #
    # predictor_crop, cfg_crop = get_predictor_crop(cfg_file=cfg_file_MSMFormer_crop,
    #                                               weight_path=weight_path_MSMFormer_crop)
    #
    # # Example of predicting and visualizing samples from OCID and OSD dataset
    # # metrics, metrics_refined = test_sample_crop(cfg, ocid_dataset[10], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)
    # test_sample_crop(cfg, osd_dataset[5], predictor, predictor_crop, visualization=True, topk=False, confident_score=0.7, print_result=True)

    # one stage model testing
    # test_dataset(cfg, pushing_dataset, predictor)
    # test_dataset(cfg, osd_dataset, predictor)
    # test_dataset(cfg, ocid_dataset, predictor)
    # test_sample(cfg, pushing_dataset[0], predictor, visualization=True)

    # Uncomment to predict a series of samples
    # met_all = []
    # met_refined_all= []
    # for i in range(1100, 1110,1):
    # # for i in [1560]:
    #     print(i)
    #     metrics, metrics_refined = test_sample_crop(cfg, ocid_dataset[i], predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
    #     met_all.append(metrics["Boundary F-measure"])
    #     met_refined_all.append(metrics_refined["Boundary F-measure"])
    # print("Boundary F-measure", np.mean(np.array(met_all)))
    # print("Refined Boundary F-measure", np.mean(np.array(met_refined_all)))

    # # Uncomment to predict the whole dataset (OSD/OCID)
    # test_dataset_crop(cfg, ocid_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
    # test_dataset_crop(cfg, osd_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)
    # test_dataset_crop(cfg, pushing_dataset, predictor, predictor_crop, visualization=False, topk=False, confident_score=0.7)

    #########################################################################################3
