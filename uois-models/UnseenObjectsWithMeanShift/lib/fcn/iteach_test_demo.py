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
import json
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
    cfg.USE_LoRA = False
    add_deeplab_config(cfg)
    add_meanshiftformer_config(cfg)
    cfg_file = cfg_file
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

    # Always use pretrained model for 2nd stage crop model
    cfg_file_MSMFormer_crop = os.path.join(
        dirname, "../../MSMFormer/configs/crop_mixture_UCN.yaml"
    )
    weight_path_MSMFormer_crop = os.path.join(
        dirname, "../../data/checkpoints/rgbd_pretrain/crop_RGBD_pretrained.pth"
    )

    models = {
        # "pretrained": ("../../MSMFormer/configs/mixture_UCN.yaml", "../../data/rgbd_pretrain/norm_RGBD_pretrained.pth"), # Pretrained model
        # "train-0": ("../../MSMFormer/human_play_rgbd_f2_mix/config.yaml", "../../MSMFormer/human_play_rgbd_f2_mix/model_0001299.pth"), # Trained model from 0 using iteach-uois training set + tod mixture
        # "finetuned": ("../../MSMFormer/human_play_rgbd_f2_mix_2120_250/config.yaml", "../../MSMFormer/human_play_rgbd_f2_mix_2120_250/model_0000999.pth"), # Finetuned model using iteach-uois training set + tod mixture and intialiazed with pretrained weights
        # "3": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s3/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s3/model_0000249.pth"),
        # "6": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s6/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s6/model_0000499.pth"),
        # "9": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s9/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s9/model_0000499.pth"),
        # "12": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s12/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s12/model_0001999.pth"),
        # "15": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s15/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s15/model_0000749.pth"),
        # "20": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s20/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s20/model_0001749.pth"),
        # "25": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s25/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s25/model_0000249.pth"),
        # "30": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s30/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s30/model_0000499.pth"),
        # "35": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s35/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s35/model_0000249.pth"),
        # "40": ("../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s40/config.yaml", "../../MSMFormer/scene_ablation_human_play_rgbd_f2_mix_s40/model_0000249.pth"),
    }

    _models = {
        "iteach-uois.latest.s1": (
            "../../MSMFormer/iteach-uois.latest.s1/config.yaml",
            "../../MSMFormer/iteach-uois.latest.s1/model_final.pth",
        ),
        "iteach-uois.latest.s2": (
            "../../MSMFormer/iteach-uois.latest.s2/config.yaml",
            "../../MSMFormer/iteach-uois.latest.s2/model_final.pth",
        ),
        "iteach-uois.latest.s3": (
            "../../MSMFormer/iteach-uois.latest.s3/config.yaml",
            "../../MSMFormer/iteach-uois.latest.s3/model_final.pth",
        ),
        "iteach-uois.latest.s4": (
            "../../MSMFormer/iteach-uois.latest.s4/config.yaml",
            "../../MSMFormer/iteach-uois.latest.s4/model_final.pth",
        ),
        "iteach-uois.latest.s5": (
            "../../MSMFormer/iteach-uois.latest.s5/config.yaml",
            "../../MSMFormer/iteach-uois.latest.s5/model_final.pth",
        ),
        "iteach-finetune-s3": (
        "../../MSMFormer/iteach-finetune-s3/config.yaml",
        "../../MSMFormer/iteach-finetune-s3/model_final.pth",
    ),
    "iteach-finetune-s6": (
        "../../MSMFormer/iteach-finetune-s6/config.yaml",
        "../../MSMFormer/iteach-finetune-s6/model_final.pth",
    ),
    "iteach-finetune-s9": (
        "../../MSMFormer/iteach-finetune-s9/config.yaml",
        "../../MSMFormer/iteach-finetune-s9/model_final.pth",
    ),
    }

    _models = {
        "iteach-finetune-s12": (
        "../../MSMFormer/iteach-finetune-s12/config.yaml",
        "../../MSMFormer/iteach-finetune-s12/model_final.pth",
    ),
    "iteach-finetune-s15": (
        "../../MSMFormer/iteach-finetune-s15/config.yaml",
        "../../MSMFormer/iteach-finetune-s15/model_final.pth",
    ),
    "iteach-finetune-s20.12": (
        "../../MSMFormer/iteach-finetune-s20.12/config.yaml",
        "../../MSMFormer/iteach-finetune-s20.12/model_final.pth",
    ),
    "iteach-finetune-s25.20": (
        "../../MSMFormer/iteach-finetune-s25.20/config.yaml",
        "../../MSMFormer/iteach-finetune-s25.20/model_final.pth",
    ),
    "iteach-finetune-s30.20": (
        "../../MSMFormer/iteach-finetune-s30.20/config.yaml",
        "../../MSMFormer/iteach-finetune-s30.20/model_final.pth",
    ),
    "iteach-finetune-s35.20": (
        "../../MSMFormer/iteach-finetune-s35.20/config.yaml",
        "../../MSMFormer/iteach-finetune-s35.20/model_final.pth",
    ),
    "iteach-finetune-s40.35": (
        "../../MSMFormer/iteach-finetune-s40.35/config.yaml",
        "../../MSMFormer/iteach-finetune-s40.35/model_final.pth",
    ),
    "iteach-finetune-s44.35": (
        "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
        "../../MSMFormer/iteach-finetune-s44.35/model_final.pth",
    ),
    }

    models = {
        # "iteach-finetune-s44.35-0000249": (
        #     "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
        #     "../../MSMFormer/iteach-finetune-s44.35/model_0000249.pth",
        # ),
        # "iteach-finetune-s44.35-0000499": (
        #     "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
        #     "../../MSMFormer/iteach-finetune-s44.35/model_0000499.pth",
        # ),
        # "iteach-finetune-s44.35-0000749": (
        #     "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
        #     "../../MSMFormer/iteach-finetune-s44.35/model_0000749.pth",
        # ),
        # "iteach-finetune-s44.35-0000999": (
        #     "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
        #     "../../MSMFormer/iteach-finetune-s44.35/model_0000999.pth",
        # ),
        "iteach-finetune-s44.35-0001249": (
            "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
            "../../MSMFormer/iteach-finetune-s44.35/model_0001249.pth",
        ),
        "iteach-finetune-s44.35-00001499": (
            "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
            "../../MSMFormer/iteach-finetune-s44.35/model_0001499.pth",
        ),
        "iteach-finetune-s44.35-0001749": (
            "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
            "../../MSMFormer/iteach-finetune-s44.35/model_0001749.pth",
        ),
        "iteach-finetune-s44.35-0001999": (
            "../../MSMFormer/iteach-finetune-s44.35/config.yaml",
            "../../MSMFormer/iteach-finetune-s44.35/model_0001999.pth",
        ),
    }


    # _results_dir = ["iteach-uois.latest.s1", "iteach-uois.latest.s2", "iteach-uois.latest.s3", "iteach-uois.latest.s4","iteach-uois.latest.s5"]
    # _results_dir = [
    #     "iteach-finetune-s3",
    #     "iteach-finetune-s6",
    #     "iteach-finetune-s9",
    #     "iteach-finetune-s12",
    #     "iteach-finetune-s15",
    #     "iteach-finetune-s20.12",
    #     "iteach-finetune-s25.20",
    #     "iteach-finetune-s30.20",
    #     "iteach-finetune-s35.20",
    #     "iteach-finetune-s40.35",
    # ]

    _results_dir = [
        # "iteach-final.s2",
        "iteach-final.s3",
        # "iteach-finetune-s6",
        # "iteach-finetune-s9",
        # "iteach-finetune-s12",
        # "iteach-finetune-s15",
        # "iteach-finetune-s20.12",
        # "iteach-finetune-s25.20",
        # "iteach-finetune-s30.20",
        # "iteach-finetune-s35.20",
        # "iteach-finetune-s40.35",
    ]

    _models = [
        "0000249",
        "0000499",
        "0000749",
        "0000999",
        "0001249",
        "0001499",
        "0001749",
        "0001999",
    ]

    models = {}
    
    _results_dir = [sys.argv[1]]

    for _result_dir in _results_dir:
        for m in _models:
            models[f"{_result_dir}-{m}"] = (
                f"../../MSMFormer/{_result_dir}/config.yaml",
                f"../../MSMFormer/{_result_dir}/model_{m}.pth",
            )

    _test_datasets = {
        "pushing": PushingDataset("test"),
        # "osd": OSDObject(image_set="test"),
        "iteach-uois": HumanPlayDataset("test"),
        # "ocid": OCIDDataset(image_set="test"),
    }

    # osd_dataset = OSDObject_UOAIS(image_set="test")

    #  ======================================= RGBD =================================================
    model_results = {}
    # model_results_dir = f"./best_model_results.corl25.finetuned-44.35.249-1999"
    # model_results_dir = f"./best_model_results.corl25.latest-s1-s5-249-1999"
    # model_results_dir = f"./best_model_results.corl25.finetuned-s3-s40.249-1999"
    model_results_dir = f"./best_model_results.corl25.iteach-reserach-s3-s40"

    os.makedirs(model_results_dir, exist_ok=True)

    for model in models:
        cfg_file, weight_path = models[model]
        cfg_file_MSMFormer = os.path.join(dirname, cfg_file)
        weight_path_MSMFormer = os.path.join(dirname, weight_path)

        print("******************************** START ********************************")
        print(cfg_file_MSMFormer)
        print(weight_path_MSMFormer)

        dataset_results = {}

        for dataset, dataset_loader in _test_datasets.items():

            try:
                predictor, cfg = get_predictor(
                    cfg_file=cfg_file_MSMFormer,
                    weight_path=weight_path_MSMFormer,
                )
            except Exception as e:
                print(f"Error loading predictor for {model}: {e}")
                continue
            
            stage1_results = test_dataset(cfg, dataset_loader, predictor)

            # predictor_crop, cfg_crop = get_predictor_crop(
            #     cfg_file=cfg_file_MSMFormer_crop, weight_path=weight_path_MSMFormer_crop
            # )

            # stage2_results = test_dataset_crop(
            #     cfg,
            #     dataset_loader,
            #     predictor,
            #     predictor_crop,
            #     visualization=False,
            #     topk=False,
            #     confident_score=0.7,
            # )

            dataset_results["model"] = model
            dataset_results["cfg"] = cfg_file_MSMFormer
            dataset_results["ckpt_path"] = weight_path_MSMFormer
            dataset_results[dataset] = {
                "results": stage1_results,
                # "results_refined": stage2_results,
            }

            json_path = os.path.join(model_results_dir, f"{model}-{dataset}.json")
            with open(json_path, "w") as f:
                json.dump(dataset_results, f, indent=4)

        # model_results[model] = dataset_results

        print("********************************  END  ********************************")
