import logging
import os
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from cv2 import imwrite
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.tools as tools
from datasets.flyingchairs import FlyingChairsValid
from datasets.flyingchairsOcc import FlyingChairsOccValid
from datasets.kitti_combined import KittiComb2015Test
from datasets.sintel import (SintelTestClean, SintelTestFinal,
                             SintelTrainingCleanFull, SintelTrainingCleanValid,
                             SintelTrainingFinalFull, SintelTrainingFinalValid)
from models.i_pmca.i_pmca import i_pmca_p

from utils.checkpoint_saver import resume_from_official
from utils.opt_flow import flow_to_image, writeFlow


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

def test(args,logger):

    # DataLoader
    with logger.LoggingBlock("Datasets", emph=True):
        # GPU parameters -- turning off pin_memory? for resolving the deadlock?
        gpuargs = {"num_workers": args.num_workers, "pin_memory": True} if args.cuda else {}
        
        # Validation dataset
        val_set = eval(args.validation_dataset)(args=args,root=args.validation_dataset_root)

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
        model = eval(args.model)(args)

        if args.cuda:
            model = torch.nn.DataParallel(model).eval()
            model = model.to('cuda')
        else:
            model = model.to('cpu')

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
                    
                    if args.cuda:
                        # Get input and target tensor keys
                        input_keys  = list(filter(lambda x: "input" in x, data.keys()))
                        tensor_keys = input_keys 
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
                        data["input1"] = nn.functional.interpolate(data["input1"] ,size=(H_,W_),mode='bilinear',align_corners=True)
                        data["input2"] = nn.functional.interpolate(data["input2"] ,size=(H_,W_),mode='bilinear',align_corners=True)

                    # Forward Pass
                    output_dict = model(data)

                    if need_resize:
                        output_dict['flow'] = nn.functional.interpolate(output_dict['flow'] ,size=(H,W),mode='bilinear',align_corners=True)
                        output_dict['flow'][0][0] = output_dict['flow'][0][0] *(W/W_)
                        output_dict['flow'][0][1] = output_dict['flow'][0][1] *(H/H_)

                        if 'occ' in output_dict:
                            output_dict['occ'] = nn.functional.interpolate(output_dict['occ'] ,size=(H,W),mode='bilinear',align_corners=True)

                    # from cv2 import imshow, waitKey
                    # from utils.opt_flow import flow_to_image
                    # imshow("im1",tools.tensor2numpy(data['input1'][0])[...,::-1])
                    # imshow("im2",tools.tensor2numpy(data['input2'][0])[...,::-1])
                    # imshow("gt",flow_to_image(tools.tensor2numpy(data['target1'][0]))[...,::-1])
                    # imshow("out",flow_to_image(tools.tensor2numpy(output_dict['flow'][0]))[...,::-1])
                    # waitKey(0)

                    # if args.model == "ds3m2_v1":
                    #     output_dict['flow'] = output_dict['flow'] * 4

                    # Save
                    flo_path = f"{args.test_output_path}/flo/{data['basedir'][0]}"
                    png_path = f"{args.test_output_path}/flo_png/{data['basedir'][0]}"
                    os.makedirs(flo_path, mode=0o777, exist_ok=True)
                    os.makedirs(png_path, mode=0o777, exist_ok=True)
                    writeFlow(f"{flo_path}/{data['basename'][0]}.flo", tools.tensor2numpy(output_dict['flow'][0]))
                    imwrite(f"{png_path}/{data['basename'][0]}.jpg", flow_to_image(tools.tensor2numpy(output_dict['flow'][0]))[...,::-1])

                    if args.test_save_occ:
                        occ_path = f"{args.test_output_path}/occ_png/{data['basedir'][0]}"
                        os.makedirs(occ_path, mode=0o777, exist_ok=True)

                        occ_ori_path = f"{args.test_output_path}/occ_ori_png/{data['basedir'][0]}"
                        os.makedirs(occ_ori_path, mode=0o777, exist_ok=True)
                        occ_pmc_path = f"{args.test_output_path}/occ_pmc_png/{data['basedir'][0]}"
                        os.makedirs(occ_pmc_path, mode=0o777, exist_ok=True)
                        occ_context_path = f"{args.test_output_path}/occ_context_png/{data['basedir'][0]}"
                        os.makedirs(occ_context_path, mode=0o777, exist_ok=True)
                        occ_no_round_path = f"{args.test_output_path}/occ_no_round_png/{data['basedir'][0]}"
                        os.makedirs(occ_no_round_path, mode=0o777, exist_ok=True)
                        occ_no_round_no_sig_path = f"{args.test_output_path}/occ_no_round_no_sig_png/{data['basedir'][0]}"
                        os.makedirs(occ_no_round_no_sig_path, mode=0o777, exist_ok=True)


                        output_occ = np.round(
                        nn.Sigmoid()(output_dict["occ"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose([0, 2, 3, 1])) * 255
                        imwrite(f"{occ_path}/{data['basename'][0]}.jpg", output_occ[0])

                        # from cv2 import imshow, waitKey,resize
                        # np_temp = nn.Sigmoid()(output_dict["hook_ori"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose([0, 2, 3, 1])[0]
                        # np_temp = np_temp - np_temp.min()
                        # np_temp = resize(np_temp,dsize=(0,0),fx=4,fy=4)
                        # np_temp = ((np_temp / np_temp.max()) *255.0).astype(np.uint8)
                        # imwrite(f"{occ_ori_path}/{data['basename'][0]}.jpg", np_temp)
                        
                        # # imshow("hook_ori",np_temp)

                        # np_temp = output_dict["hook_pmc"].expand(-1, 3, -1, -1).data.cpu().numpy().transpose([0, 2, 3, 1])[0]
                        # np_temp = np_temp - np_temp.min()
                        # np_temp = resize(np_temp,dsize=(0,0),fx=4,fy=4)
                        # np_temp = ((np_temp / np_temp.max()) *255.0).astype(np.uint8)
                        # imwrite(f"{occ_pmc_path}/{data['basename'][0]}.jpg", np_temp)

                        # # imshow("hook_pmc",np_temp)

                        # np_temp = output_dict["hook_context"].expand(-1, 3, -1, -1).data.cpu().numpy().transpose([0, 2, 3, 1])[0]
                        # np_temp = np_temp - np_temp.min()
                        # np_temp = resize(np_temp,dsize=(0,0),fx=4,fy=4)
                        # np_temp = ((np_temp / np_temp.max()) *255.0).astype(np.uint8)
                        # imwrite(f"{occ_context_path}/{data['basename'][0]}.jpg", np_temp)

                        # # imshow("hook_context",np_temp)

                        # np_temp = output_dict["occ"].expand(-1, 3, -1, -1).data.cpu().numpy().transpose([0, 2, 3, 1])[0]
                        # np_temp = np_temp - np_temp.min()
                        # # np_temp = resize(np_temp,dsize=(0,0),fx=4,fy=4)
                        # np_temp = ((np_temp / np_temp.max()) *255.0).astype(np.uint8)
                        # imwrite(f"{occ_no_round_no_sig_path}/{data['basename'][0]}.jpg", np_temp)

                        # # imshow("occ_no_round_no_sig",np_temp)

                        np_temp = nn.Sigmoid()(output_dict["occ"]).expand(-1, 3, -1, -1).data.cpu().numpy().transpose([0, 2, 3, 1])[0]
                        np_temp = np_temp - np_temp.min()
                        # np_temp = resize(np_temp,dsize=(0,0),fx=4,fy=4)
                        np_temp = ((np_temp / np_temp.max()) *255.0).astype(np.uint8)
                        imwrite(f"{occ_no_round_path}/{data['basename'][0]}.jpg", np_temp)

                        # # imshow("occ_no_round",np_temp)

                        # # waitKey(0)

                        # # print(233)