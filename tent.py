import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from itertools import chain

import parse_args
from datasets.utils import get_dataset, get_train_dataset
from models.utils import get_model
from trainers.utils import get_trainer
from utils import basics
from wrappers.utils import get_warpped_model
import wandb
from open_clip import create_model_from_pretrained
from prompt_learner import ClipPromptClassifier

#os.environ["WANDB_DISABLED"] = "true"


def create_exerpiment_setting(args):
    # get hash
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.lr = args.blr

    args.save_folder = os.path.join(
        args.exp_path,
        "tent",
        args.dataset,
        "biomed_clip",
        args.sensitive_name,
        f"seed{args.random_seed}",
    )

    args.resume_path = args.save_folder
    basics.creat_folder(args.save_folder)

    try:
        with open(f"configs/datasets/{args.dataset}.json", "r") as f:
            data_setting = json.load(f)
            data_setting["augment"] = False
            data_setting["test_meta_path"] = data_setting[
                f"test_{str.lower(args.sensitive_name)}_meta_path"]
            args.data_setting = data_setting

            if args.pos_class is not None:
                args.data_setting["pos_class"] = args.pos_class
    except:
        args.data_setting = None

    return args

def train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset",
        default="CXP",
        choices=[
            "CXP",
            "MIMIC_CXR",
            "HAM10000",
            "PAPILA",
            "ADNI",
            "COVID_CT_MD",
            "FairVLMed10k",
            "BREST",
            "GF3300",
            "HAM10000-Seg",
            "FairSeg",
            "montgomery",
            "TUSC"
        ],
    )
    parser.add_argument("--sensitive_name", default="Sex",
                        choices=["Sex", "Age", "Race", "Language"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--exp_path", type=str, default="./output")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                         help='initial learning rate', dest='lr')
    parser.add_argument('-plr', '--prompt-lr', default=5e-4, type=float,
                        help='prompt learning rate (default: 5e-4)')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    parser.add_argument("--total_epochs", type=int,
                        default=100, help="total training epochs")
    parser.add_argument("--no_cuda", dest="cuda", action="store_false")

    args = parser.parse_args()
    args = create_exerpiment_setting(args)

    logger = basics.setup_logger(
        "train", args.save_folder, "no_oracle_n_pseudo_label_per_group_confidence_cls_spu", screen=True, tofile=True) # history.log oracle_pseudo_labels 0.5_n_pseudo_label_per_group_confidence_cls
    logger.info("Using following arguments for training.")
    logger.info(args)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataloader, val_dataloader, train_meta = get_train_dataset(args, split="train")
    test_data, test_dataloader, test_meta = get_dataset(args, split="test")

    model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )
    model = get_model(args).to(args.device)

    classifier = ClipPromptClassifier(model=model, class_names=args.data_setting["class_names"], 
                                      device=args.device, template=args.data_setting["text_template"]) 
    
    optimizer = torch.optim.Adam(classifier.get_parameters(prompt_lr=args.prompt_lr), args.lr, weight_decay=args.wd, nesterov=True)
    #lr_scheduler = torch.optim.lr_scheduler.Cosine(optimizer, step_size=1, gamma=args.lr_decay)