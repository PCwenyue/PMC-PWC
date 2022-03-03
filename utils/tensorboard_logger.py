import glob
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import utils.tools as tools
from models.pwc_modules import upsample2d_as
from utils.opt_flow import flow_to_image

class tensorboard_Logger(object):
    def __init__(self, args):
        self.args = args

        # if args.checkpoint.endswith(".pt"):
        self.filepath = f"{args.log_path}/tb_log"
        # else:
        #     self.filepath = f"{args.checkpoint}/tb_log"

        try:
            os.makedirs(f'{self.filepath}/train')
            os.makedirs(f'{self.filepath}/val')
            os.makedirs(f'{self.filepath}/imgs')
        except:
            pass

        self.train_writer = SummaryWriter(log_dir=f"{self.filepath}/train")
        self.val_writer   = SummaryWriter(log_dir=f"{self.filepath}/val")

        
    
    def log(self, log_type, scalar_dict, input_dict, output_dict, lr, step, write_img=False):
        assert log_type in ['train', 'val']
        if log_type is 'train':
            writer = self.train_writer
        else:
            writer = self.val_writer

        for item in scalar_dict:
            if item.endswith("loss"):
                scalar_type = "loss"
            elif item == "F1":
                scalar_type = "F1"
            elif item == "epe":
                scalar_type = "epe"
            writer.add_scalar(f"{scalar_type}/{item}",scalar_dict[f"{item}"],step)
        
        if input_dict is not None and output_dict is not None:

            img_pair = torch.cat((input_dict['input1'][0],input_dict['input2'][0]),2)

            flo_f_input  = torch.from_numpy(flow_to_image(tools.tensor2numpy(input_dict['target1'][0])).transpose(2,0,1))
            if log_type is 'train':
                if len(output_dict['flow']) ==5:
                    flo_f_output = torch.from_numpy(flow_to_image(tools.tensor2numpy(upsample2d_as(output_dict['flow'][len(output_dict['flow'])-1], input_dict['target1'], mode="bilinear"))[0] * (1.0 / self.args.div_flow)).transpose(2,0,1))
                else:
                    flo_f_output = torch.from_numpy(flow_to_image(tools.tensor2numpy(upsample2d_as(output_dict['flow'][6][0], input_dict['target1'], mode="bilinear"))[0] * (1.0 / self.args.div_flow)).transpose(2,0,1))
            else:
                flo_f_output = torch.from_numpy(flow_to_image(tools.tensor2numpy(output_dict['flow'][0])).transpose(2,0,1))
            flo_f_pair = (torch.cat((flo_f_input,flo_f_output),2).float()/255.0).to('cuda')

            log_img = torch.cat((img_pair,flo_f_pair),1)

            if 'occ' in output_dict:

                occ_f_input_np  = input_dict['target_occ1'].expand(-1, 3, -1, -1).data.cpu().numpy()

                if log_type is 'train':
                    occ_f_output = upsample2d_as(output_dict['occ'][6][0], input_dict['target_occ1'], mode="bilinear")
                else:
                    occ_f_output = output_dict['occ']
                
                occ_f_output_np = np.round(
                nn.Sigmoid()(occ_f_output).expand(-1, 3, -1, -1).data.cpu().numpy())
                
                occ_f_pair = torch.from_numpy(np.concatenate((occ_f_input_np[0],occ_f_output_np[0]), axis=2)).to('cuda')

                log_img = torch.cat((log_img,occ_f_pair),1)

            writer.add_image(tag=f'{log_type}',img_tensor=log_img,global_step=step)
            if write_img:
                write_img = (transforms.ToPILImage()(log_img.detach().cpu())).convert('RGB')
                name = input_dict['basename'][0]
                epe = scalar_dict['epe']

                if 'F1' in scalar_dict:
                    F1 = scalar_dict['F1']

                    write_img.save(f'{self.filepath}/imgs/{step}-{name}-epe-{epe:0.4f}-F1-{F1:0.4f}.jpg')
                else:
                    write_img.save(f'{self.filepath}/imgs/{step}-{name}-epe-{epe:0.4f}.jpg')
                    
        if lr is not None:
            writer.add_scalar("lr",lr,step)
