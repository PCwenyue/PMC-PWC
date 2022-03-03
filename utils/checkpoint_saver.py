import glob
import os
import shutil
import time

import torch
import logging

def resume_from_official(args, model, path=None):
    
    # logging.info(f"resume_type: resume_from_official")

    if path.endswith('pwc_net_nvidia_after_sintel.pth.tar'):
        if args.cuda:
            ckpt = torch.load(path)
        else:
            ckpt = torch.load(path, map_location=torch.device('cpu'))
        c2n = {'a':0, 'aa':1, 'b':2}
        c2n_s = {'aa':0, 'a':1, 'b':2}

        for item in ckpt:
            if args.cuda:
                string = "model.module"
            else:
                string = "model"
            keys = item.split('.')
            if keys[0].endswith('a') or keys[0].endswith('aa') or keys[0].endswith('b'):
                main_num = int(keys[0].split('conv')[1][0])-1

                if main_num == 5:
                    sub_num  = c2n_s[keys[0].split('conv')[1][1:]]
                else:
                    sub_num  = c2n[keys[0].split('conv')[1][1:]]

                string = f"{string}.feature_pyramid_extractor.convs._modules['{main_num}']._modules['{sub_num}']._modules['0'].{keys[-1]}"
            
            elif keys[0].endswith('_0') or keys[0].endswith('_1') or keys[0].endswith('_2') or keys[0].endswith('_3') or keys[0].endswith('_4') or keys[0].endswith('_5') or keys[0].endswith('_6'):
                main_num = 6 - int(keys[0].split('conv')[1][0])
                sub_num  = int(keys[0].split('conv')[1][2])+1
                string = f"{string}.flow_estimators._modules['{main_num}'].conv{sub_num}._modules['0'].{keys[-1]}"

            elif keys[0][0] == 'p':
                main_num = 6 - int(keys[0].split('flow')[1][0])
                string = f"{string}.flow_estimators._modules['{main_num}'].conv_last._modules['0'].{keys[-1]}"
            
            elif keys[0][0:2] == 'de':
                main_num = 6 - int(keys[0].split('deconv')[1][0])
                string = f"{string}.upflow._modules['{main_num}'].{keys[-1]}"

            elif keys[0][0] == 'u':
                main_num = 6 - int(keys[0].split('upfeat')[1][0])
                string = f"{string}.upfeat._modules['{main_num}'].{keys[-1]}"

            elif keys[0][0:2] == 'dc':
                main_num = int(keys[0].split('conv')[1][0]) - 1
                string = f"{string}.context_networks.convs._modules['{main_num}']._modules['0'].{keys[-1]}"

            if keys[0][0:2] == 'dc' and keys[-1] == 'weight' and main_num == 0 and args.model!= 'pwcnet_plus':
                try:
                    if eval(string).data[:,:-2,:,:].shape == ckpt[item].shape:
                        # eval(string).data[:,:-2,:,:] = ckpt[item].to('cuda')
                        eval(string).data[:,:-2,:,:] = ckpt[item].type_as(eval(string).data)
                    else:
                        logging.info(f"{string} pass")
                        pass
                except:
                    logging.info(f"{string} does not exist")    
            else:
                try:
                    if eval(string).data.shape == ckpt[item].shape:
                        # eval(string).data = ckpt[item].to('cuda')
                        eval(string).data = ckpt[item].type_as(eval(string).data)
                    else:
                        logging.info(f"{string} pass")
                        pass
                except:
                    logging.info(f"{string} does not exist")

    elif 'RAFT' in path:
        model.load_state_dict(torch.load(path))

    else:
        if args.cuda:
            ckpt = torch.load(path)['state_dict']
        else:
            ckpt = torch.load(path, map_location=torch.device('cpu'))['state_dict']
        

        for item in ckpt:
            keys = item.split('_model.')[1].split('.')
            if args.cuda:
                string = "model.module"
            else:
                string = "model"
            for idx in range(len(keys)):
                key = keys[idx]
                if key.isdigit():
                    string = f"{string}._modules['{keys[idx]}']"
                else:
                    string = f"{string}.{keys[idx]}"
            try:
                if eval(string).data.shape == ckpt[item].shape:
                    eval(string).data = ckpt[item]
                else:
                    print(f"{string} pass")
                    pass
            except:
                logging.info(f"{string} does not exist")
    ckpt = []

class Saver(object):
    def __init__(self, args):
        self.max_to_keep = args.checkpoint_max_to_keep
        self.args = args

        self.filepath = f"{args.log_path}/weights"
        
    def save_checkpoint(self, model, optimizer, scheduler, input_epe, epoch):
        is_best = False
        epe_list = []
        input_epe_np = input_epe

        try:
            os.makedirs(f'{self.filepath}')
        except:
            pass

        all_ckpt = glob.glob(f"{self.filepath}/*.pt")
        if len(all_ckpt) < self.max_to_keep:
           is_best = True
        else:
            for ckpt in all_ckpt:
                _, epe = ckpt.split(f"{self.filepath}/")[1].split(".pt")[0].split("-epe")
                epe_float = float(epe)
                epe_list.append(epe_float)

                if input_epe_np < epe_float:
                    is_best = True

            index = epe_list.index(max(epe_list))
            os.remove(all_ckpt[index])
        
        filename = f"{self.filepath}/{epoch}-epe{input_epe:0.4f}.pt"
        state = {'epoch': epoch,
                 'model_state_dict': model.module.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict()}#,
                # 'scheduler_state_dict':scheduler.state_dict()}
                 

        torch.save(state, filename)  # save checkpoint
        if is_best:
            logging.info(f"=> Saving a new best epe {input_epe:0.4f}\n")
        else:
            logging.info("=> Validation EPE did not improve, saving up-to-data result\n")

    def resume_from_checkpoint(self, args, model, optimizer, scheduler, resume_type, path=None):
        assert resume_type in ["resume_from_pretraining", "resume_from_latest", "resume_from_best","training_from_scratch","resume_from_official"]
        
        logging.info(f"resume_type: {resume_type}")

        if resume_type is "resume_from_pretraining":
            if args.cuda:
                checkpoint = torch.load(args.checkpoint)
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
                checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))['state_dict']
                model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint = []
            return 1

        elif resume_type is "training_from_scratch":
            return 1
        elif resume_type is "resume_from_official":
            resume_from_official(self.args, model, path)
            return 1
        else:
            all_ckpt = glob.glob(f"{self.filepath}/*.pt")
            
            epoch_list = []
            epe_list = []
            for ckpt in all_ckpt:
                epoch, epe = ckpt.split(f"{self.filepath}/")[1].split(".pt")[0].split("-epe")
                epoch_list.append(int(epoch))
                epe_list.append(float(epe))
            if resume_type is 'resume_from_latest':
                index = epoch_list.index(max(epoch_list))
            else:
                index = epe_list.index(min(epe_list))

            checkpoint = torch.load(all_ckpt[index])
            if args.cuda:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 更新学习率
            scheduler.last_epoch = checkpoint['epoch']
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()

            checkpoint = []

            #return checkpoint['epoch'] + 1
            return 28


