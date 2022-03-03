import logging
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.tools as tools
from datasets.flyingchairs import FlyingChairsFull, FlyingChairsValid
from datasets.flyingchairsOcc import FlyingChairsOccFull, FlyingChairsOccValid
from datasets.sintel import (SintelTrainingCleanFull, SintelTrainingCleanValid,
                             SintelTrainingCombFull, SintelTrainingCombValid,
                             SintelTrainingFinalFull, SintelTrainingFinalValid)
from datasets.kitti_combined import KittiComb2015Train
from models.i_pmca.i_pmca import i_pmca_p
# from models.IRR_PWC.IRR_PWC import IRR_PWC
from models.losses import (PSNR, SSIM, MultiScaleEPE_PWC,
                           MultiScaleEPE_PWC_Bi_Occ_upsample,
                           MultiScaleEPE_PWC_Bi_Occ_upsample_edge,
                           MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel_edge,
                           MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI,
                           MultiScaleEPE_PWC_edge, 
                           sequence_loss,sequence_loss_occ,sequence_loss_occ_sintel)

from utils.checkpoint_saver import resume_from_official


def _sizes_to_str(value):
            if np.isscalar(value):
                return '[1L]'
            else:
                return ' '.join([str([d for d in value.size()])])

def _log_statistics(logger, dataset, prefix, name):
    with logger.LoggingBlock(f"{prefix} Dataset: {name}"):
        example_dict = dataset[0]  # get sizes from first dataset example
        for key, value in sorted(example_dict.items()):
            if key in ["index", "basename"]:  # no need to display these
                continue
            if isinstance(value, str):
                logging.info(f"{key}: {value}")
            else:
                logging.info(f"{key}: {_sizes_to_str(value)}")
        logging.info(f"num_examples: {len(dataset)}")

def tensor2float_dict(tensor_dict):
    return {key: tensor.item() for key, tensor in tensor_dict.items()}

def evaluation(args,logger):

    # DataLoader
    with logger.LoggingBlock("Datasets", emph=True):
        # GPU parameters -- turning off pin_memory? for resolving the deadlock?
        gpuargs = {"num_workers": args.num_workers, "pin_memory": False} if args.cuda else {}
        
        # Validation dataset
        val_set   = eval(args.validation_dataset)(args=args,root=args.validation_dataset_root, photometric_augmentations=False)

        # Create validation dataset
        validation_loader = DataLoader(
            dataset=val_set,
            batch_size=args.batch_size_val,
            **gpuargs)

        # Check dataloader
        success = any(loader is not None for loader in [validation_loader])
        if not success:
            logging.info("No dataset could be loaded successfully. Please check dataset paths!")
            quit()
        
        _log_statistics(logger, val_set,   prefix="Validation", name=args.validation_dataset)
    

    # Model
    with logger.LoggingBlock("Model", emph=True):
        logging.info(f"{args.model}")
        model = eval(args.model)(args).eval()
        validation_loss = eval(args.validation_loss)(args).eval()

        if args.cuda:
            model = model.to('cuda')
            model = torch.nn.DataParallel(model).eval()
            validation_loss = validation_loss.to('cuda')
        else:
            model = model.to('cpu')
            validation_loss = validation_loss.to('cpu')

        logging.info(f"parameters: {sum(param.numel() for param in model.parameters())}")

        if args.checkpoint.endswith(".ckpt") or args.checkpoint.endswith(".pth.tar") or ('RAFT_offcially' in args.checkpoint):
            resume_from_official(args, model, args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint)
            if args.cuda:
                model_dict=model.module.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
                for k, v in checkpoint['model_state_dict'].items():
                    if k not in model_dict:
                        print(f"{k} is not exist in model")
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                model.module.load_state_dict(model_dict)
                # model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

        logging.info(f"Loaded ckpt from {args.checkpoint}")

    # Cuda optimization   
    with logger.LoggingBlock("Cudnn", emph=True): 
        if args.cuda:
            torch.backends.cudnn.benchmark = False
        logging.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    
    # Evaluation
    with logger.LoggingBlock("Evaluation", emph=True):
        logging.info(f"start_epoch: {args.start_epoch}")
        logging.info(f"total_epochs: {args.total_epochs}")

    # rescale the image size to be multiples of 64
    divisor = 64.

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        # Validation
        # Keep track of moving averages
        moving_averages_dict_val = None
        with tqdm(validation_loader,desc=f"Epoch {epoch}/{args.total_epochs}",dynamic_ncols=True,ascii=True,miniters=1) as loader:
            with torch.no_grad():
                model.eval()
                for i, data in enumerate(loader):
                    input_data = {}
                    if args.cuda:
                        # Get input and target tensor keys
                        input_keys  = list(filter(lambda x: "input" in x, data.keys()))
                        target_keys = list(filter(lambda x: "target" in x, data.keys()))
                        tensor_keys = input_keys + target_keys
                        # Possibly transfer to Cuda
                        for key, value in data.items():
                            if key in tensor_keys:
                                data[key] = value.cuda(non_blocking=False)
                    
                    # Extract batch size from first input
                    batch_size = data["input1"].size()[0]
                    
                    H = data["input1"].size()[2]
                    W = data["input1"].size()[3]
                    need_resize = False

                    if H % divisor != 0:
                        H_ = int(ceil(H/divisor) * divisor)
                        need_resize = True
                    else:
                        H_ = H

                    if W % divisor != 0:
                        W_ = int(ceil(W/divisor) * divisor)
                        need_resize = True
                    else:
                        W_ = W
                    
                    if need_resize:
                        input_data["input1"] = nn.functional.interpolate(data["input1"] ,size=(H_,W_),mode='bilinear',align_corners=True)
                        input_data["input2"] = nn.functional.interpolate(data["input2"] ,size=(H_,W_),mode='bilinear',align_corners=True)
                    else:
                        input_data["input1"] = data["input1"]
                        input_data["input2"] = data["input2"]
                    # Forward Pass
                    output_dict = model(input_data)

                    if need_resize:
                        output_dict['flow'] = nn.functional.interpolate(output_dict['flow'] ,size=(H,W),mode='bilinear',align_corners=True)
                        output_dict['flow'][0][0] = output_dict['flow'][0][0] *(W/W_)
                        output_dict['flow'][0][1] = output_dict['flow'][0][1] *(H/H_)

                        if 'occ' in output_dict:
                            output_dict['occ'] = nn.functional.interpolate(output_dict['occ'] ,size=(H,W),mode='bilinear',align_corners=True)
                    
                    # if args.model == "ds3m2_v1":
                    #     output_dict['flow'] = output_dict['flow'] * 4

                    # Compute Loss
                    loss_dict = validation_loss(output_dict,data)

                    # Convert loss dictionary to float
                    loss_dict_per_step = tensor2float_dict(loss_dict)

                    # Possibly initialize moving averages
                    if moving_averages_dict_val is None:
                        moving_averages_dict_val = {
                            key: tools.MovingAverage() for key in loss_dict_per_step.keys()
                        }

                    # Add moving average
                    for key, loss in loss_dict_per_step.items():
                        moving_averages_dict_val[key].add_average(loss, addcount=batch_size)
                        

                # Record average losses
                avg_loss_dict = { key: ma.mean() for key, ma in moving_averages_dict_val.items() }
                # logging.info(f"epe: {avg_loss_dict['epe']:.4f}, F1: {avg_loss_dict['F1']:.4f}")
                
                log_info = ""
                for item in avg_loss_dict:
                    log_info = f"{log_info}, {item}: {avg_loss_dict[item]:.4f}"
                log_info = log_info[2:]
                logging.info(log_info)
