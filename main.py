import argparse
import logging
import os
import random
import sys
import time
from test import test

import colorama
import numpy as np
import torch

import utils.logger as logger
from evaluation import evaluation
from test_KITTI import test_kitti
# from train import train
from utils.tools import (str2bool, str2intlist, str2intlist_or_none,
                         str2str_or_none, write_dictionary_to_file)


def main():
    # Change working directory    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Argument parser and shortcut function to add arguments
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    # Standard arguments
    add("--batch_size_train", type=int,       default=4)
    add("--batch_size_val",   type=int,       default=1)
    add("--num_workers",      type=int,       default=0)
    add("--seed",             type=int,       default=1)
    add("--tensorboard",      type=str2bool,  default=False) 
    add("--log_path", type=str2str_or_none,   default=None) 
    add("--tensorboard_log_step", type=int,   default=5) 
    # add("--checkpoint", type=str2str_or_none, default=None)
    add("--checkpoint", type=str2str_or_none, default="./weights/PMC_PWC_noedge_KITTI.pt")
    add("--checkpoint_mode", type=str,        default="resume_from_latest",
            choices=["resume_from_pretraining", "resume_from_latest", "resume_from_best","training_from_scratch","resume_from_official"])
    add("--checkpoint_max_to_keep", type=int, default=20)
    add("--cuda",            type=str2bool, default=True)
    # add("--evaluation",      type=str2bool, default=False)
    add("--evaluation",      type=str2bool, default=True)
    add("--test",            type=str2bool, default=False)
    # add("--test",            type=str2bool, default=True)
    add("--test_save_occ",type=str2bool,    default=True)
    # add("--test_output_path",type=str2str_or_none, default=None)
    add("--test_output_path",type=str2str_or_none, default="./debug")
    add("--start_epoch",     type=int,            default=1)
    add("--total_epochs",    type=int,            default=1)

    # Arguments inferred from losses
    add("--training_loss",   type=str2str_or_none, default="MultiScaleEPE_PWC_Bi_Occ_upsample_edge",
                 choices=[
                     "MultiScaleEPE_PWC","MultiScaleEPE_PWC_edge",
                     "MultiScaleEPE_PWC_Bi_Occ_upsample", 
                     "MultiScaleEPE_PWC_Bi_Occ_upsample_edge", 
                     "MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel", 
                     "MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel_edge", 
                     "MultiScaleEPE_PWC_KITTI","MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI",
                     "sequence_loss","sequence_loss_occ","sequence_loss_occ_sintel"])
    add("--validation_loss", type=str2str_or_none, default="MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI",
                 choices=[
                     "MultiScaleEPE_PWC","MultiScaleEPE_PWC_edge",
                     "MultiScaleEPE_PWC_Bi_Occ_upsample", 
                     "MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI",
                     "MultiScaleEPE_PWC_Bi_Occ_upsample_edge", 
                     "MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel", 
                     "MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel_edge", 
                     "MultiScaleEPE_PWC_KITTI","MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI",
                     "SSIM","PSNR",
                     "sequence_loss","sequence_loss_occ","sequence_loss_occ_sintel"])

    # Arguments inferred from models
    add("--model", type=str,      default="i_pmca_p",
        choices=[
            "pwcnet_plus","IRR_PWC",
            "RAFT","RAFT_OCC",
            "i_pmca_p","i_pmca_v2_s","i_pmca_s",
            "i_pmca_v2_p","i_pmca_lite",
            "PWCNet","pwcnet_irr"])

    add("--div_flow", type=float, default=0.05)
    
    # Arguments inferred from augmentations for training
    add("--training_augmentation_RandomAffine", type=str2str_or_none,    default="RandomAffineFlowOcc",
        choices=[
            None, 
            "RandomAffineFlow",
            "RandomAffineFlowOcc",
            "RandomAffineFlowOccSintel",
            "RandomAffineFlowOccKITTI"])

    add("--training_augmentation_crop", type=str2intlist_or_none,          default=None)
    # add("--training_augmentation_crop", type=str2intlist_or_none,          default=[320,896])
    add("--training_augmentation_ColorJitter", type=str2intlist_or_none, default=[0.5,0.5,0.5,0.5],
                                               help='[brightness, contrast, saturation, hue] or None')
    add("--training_augmentation_RandomGamma", type=str2intlist_or_none, default=[0.7,1.5,True],
                                               help='[min_gamma, max_gamma, clip_image] or None')

    # Arguments inferred from datasets for training
    add("--training_dataset", type=str,      default="FlyingChairsOccTrain",
        choices=[
            "FlyingChairsTrain",
            "FlyingChairsOccTrain", 
            "FlyingThings3dCleanTrain",
            "SintelTrainingCombFull","SintelTrainingCombTrain",
            "KittiCombFull"])
    # add("--training_dataset_root", type=str, default="/home/fengcheng970pro/Documents/dataset/KITTI")
    add("--training_dataset_root", type=str, default="/home/fengcheng970pro/Documents/dataset/FlyingChairsOcc/data")

    # Arguments inferred from datasets for validation
    add("--validation_dataset", type=str,     default="KittiComb2015Train",
        choices=[
            "FlyingChairsValid","FlyingChairsFull",
            "FlyingChairsOccValid","FlyingChairsOccFull",
            "SintelTrainingCleanFull","SintelTrainingFinalFull",
            "SintelTrainingCleanValid","SintelTrainingFinalValid",
            "SintelTestClean","SintelTestFinal",
            "SintelTrainingCombValid", "SintelTrainingCombFull",
            "KittiCombVal","KittiComb2015Train","KittiComb2015Test","KittiComb2015Full","KittiComb2012Test"])
    # add("--validation_dataset_root",type=str, default="/home/fengcheng970pro/Documents/dataset/MPI-Sintel-complete")
    # add("--validation_dataset_root",type=str, default="/home/fengcheng970pro/Documents/dataset/FlyingChairsOcc/data")
    add("--validation_dataset_root",type=str, default="/home/fengcheng970pro/Documents/dataset/KITTI")

    # Arguments inferred from PyTorch optimizers
    add("--optimizer", type=str,                default="AdamW", choices=["Adam", "AdamW"])
    add("--optimizer_lr", type=float,           default=3e-5)
    add("--optimizer_weight_decay", type=float, default=4e-4)
    add("--optimizer_amsgrad", type=str2bool,   default=False)

    # Arguments inferred from PyTorch lr schedulers
    add("--lr_scheduler", type=str,                    default="MultiStepLR")
    add("--lr_scheduler_gamma", type=float,            default=0.5)
    add("--lr_scheduler_milestones", type=str2intlist, default=[1, 2, 3])

    # Add special arguments for training
    add("--training_key", type=str, default="total_loss")

    # Add special arguments for validation
    add("--validation_key", type=str, default="epe")

    # Parse arguments
    args = parser.parse_args()

    # 后处理
    if args.training_augmentation_RandomAffine is None:
        args.training_augmentation_crop = None
    
    if args.evaluation or args.test:
        args.batch_size_val = 1
        args.num_workers = 0
        args.tensorboard = False
        args.start_epoch = 1
        args.total_epochs = 1

    # Add special arguments for checkpoints
    time_now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

    if args.checkpoint is None:
        args.checkpoint = f"experiments/{args.training_dataset}-{time_now}"
        args.checkpoint_mode = "training_from_scratch"
        log_path = args.checkpoint
        args.log_path = log_path

    elif args.checkpoint.endswith(".pt"):
        args.checkpoint_mode = "resume_from_pretraining"
        
        if args.evaluation:
            log_path = f"{args.checkpoint.split('/weights/')[0]}/evaluation/{args.validation_dataset}-{args.checkpoint.split('/weights/')[1]}"
            os.makedirs(log_path, mode=0o777, exist_ok=True)
        elif args.test:
            # log_path = f"{args.checkpoint.split('/weights/')[0]}/test/{args.validation_dataset}-{args.checkpoint.split('/weights/')[1]}"
            log_path = args.test_output_path
            os.makedirs(log_path, mode=0o777, exist_ok=True)
        else:
            log_path = f"experiments/{args.training_dataset}-{time_now}"
            args.log_path = log_path
    
    elif args.checkpoint.endswith(".ckpt") or args.checkpoint.endswith(".pth.tar") or 'RAFT' in args.checkpoint:
        args.checkpoint_mode = "resume_from_official"

        if args.evaluation:
            log_path = f"experiments/{args.validation_dataset}-{time_now}"
            args.log_path = log_path
        elif args.test:
            log_path = args.test_output_path
            os.makedirs(log_path, mode=0o777, exist_ok=True)
        else:
            log_path = f"experiments/{args.training_dataset}-{time_now}"
            args.log_path = log_path
    
    else:
        log_path = args.checkpoint
        args.log_path = log_path


    # Parse default arguments from a dummy commandline not specifying any args
    defaults = vars(parser.parse_known_args(['--dummy'])[0])

    # Consistency checks
    args.cuda = args.cuda and torch.cuda.is_available()

    # Write arguments to file, as txt
    write_dictionary_to_file(
        sorted(vars(args).items()),
        filename=os.path.join(log_path, 'args.txt'))
    
    # Setup logbook before everything else
    logger.configure_logging(os.path.join(log_path, 'logbook.txt'))

    # Log arguments
    with logger.LoggingBlock("Commandline Arguments", emph=True):
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.CYAN
            logging.info('{}{}: {}{}'.format(color, argument, value, reset))
    
    # Set random seed, possibly on Cuda
    with logger.LoggingBlock("Random Seeds", emph=True):
        # python
        seed = args.seed
        random.seed(seed)
        logging.info("Python seed: %i" % seed)
        # numpy
        seed += 1
        np.random.seed(seed)
        logging.info("Numpy seed: %i" % seed)
        # torch
        seed += 1
        torch.manual_seed(seed)
        logging.info("Torch CPU seed: %i" % seed)
        # torch cuda
        seed += 1
        torch.cuda.manual_seed(seed)
        logging.info("Torch CUDA seed: %i" % seed)

    if args.evaluation:
        evaluation(args,logger)
    elif args.test:
        if args.validation_dataset in ["KittiComb2015Test", "KittiComb2015Full", "KittiComb2012Test"]:
            test_kitti(args,logger)
        else:
            test(args,logger)
    # else:
        # train(args,logger)

if __name__ == "__main__":
    main()
