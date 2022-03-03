from __future__ import absolute_import, division, print_function

import os
from math import exp

import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.tools import tensor2numpy



def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return F.adaptive_avg_pool2d(inputs, [h, w])

def _upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return F.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)

def fbeta_score(y_true, y_pred, beta, eps=1e-8):
    beta2 = beta ** 2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=2).sum(dim=2)
    precision = true_positive / (y_pred.sum(dim=2).sum(dim=2) + eps)
    recall = true_positive / (y_true.sum(dim=2).sum(dim=2) + eps)

    return torch.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))

def f1_score_bal_loss(y_pred, y_true):
    eps = 1e-8

    tp = -(y_true * torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    fn = -((1 - y_true) * torch.log((1 - y_pred) + eps)).sum(dim=2).sum(dim=2).sum(dim=1)

    denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps
    denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps

    return ((tp / denom_tp).sum() + (fn / denom_fn).sum()) * y_pred.size(2) * y_pred.size(3) * 0.5


class MultiScaleEPE_FlowNet(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet, self).__init__()
        self._args = args        
        self._batch_size = args.batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target = self._args.div_flow * target_dict["target1"]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                target_i = _downsample2d_as(target, output_i)
                epe_i = _elementwise_epe(output_i, target_i)
                total_loss = total_loss + self._weights[i] * epe_i.sum() / self._batch_size
                loss_dict["epe%i" % (i + 2)] = epe_i.mean()
            loss_dict["total_loss"] = total_loss
        else:
            output = output_dict["flow1"]
            target = target_dict["target1"]
            epe = _elementwise_epe(output, target)
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_FlowNet_IRR(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR, self).__init__()
        self._args = args        
        self._batch_size = args.batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj, target_f_ii)
                    total_loss = total_loss + self._weights[ii] * epe_f_ii.sum()
                    loss_dict["epe%i" % (ii + 2)] = epe_f_ii.mean()
            loss_dict["total_loss"] = total_loss / self._batch_size / self._num_iters

        else:
            output = output_dict["flow1"]
            target_f = target_dict["target1"]
            epe_f = _elementwise_epe(target_f, output)
            loss_dict["epe"] = epe_f.mean()

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Bi(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR_Bi, self).__init__()
        self._args = args        
        self._batch_size = args.batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["target1"]
            target_b = self._args.div_flow * target_dict["target2"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    total_loss = total_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum())
                    loss_dict["epe%i" % (ii + 2)] = (epe_f_ii.mean() + epe_b_ii.mean()) / 2
            loss_dict["total_loss"] = total_loss / self._batch_size / self._num_iters / 2
        else:
            epe_f = _elementwise_epe(output_dict["flow1"], target_dict["target1"])
            loss_dict["epe"] = epe_f.mean()

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR_Occ, self).__init__()
        self._args = args        
        self._batch_size = args.batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

        self.f1_score_bal_loss = f1_score_bal_loss
        self.occ_activ = nn.Sigmoid()
        
    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]
            outputs_occ = [output_dict[key] for key in ["occ2", "occ3", "occ4", "occ5", "occ6"]]

            # div_flow trick
            target = self._args.div_flow * target_dict["target1"]
            target_occ = target_dict["target_occ1"]

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                target_ii = _downsample2d_as(target, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    flow_loss = flow_loss + self._weights[ii] * _elementwise_epe(output_ii_jj, target_ii).sum()

            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ, output_ii[0])
                for jj, output_ii_jj in enumerate(output_ii):
                    occ_loss = occ_loss + self._weights[ii] * self.f1_score_bal_loss(self.occ_activ(output_ii_jj), target_occ_f)

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size / self._num_iters
            loss_dict["occ_loss"] = occ_loss / self._batch_size / self._num_iters
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / self._num_iters

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow1"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ1"])))

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Bi_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_FlowNet_IRR_Bi_Occ, self).__init__()
        self._args = args        
        self._batch_size = args.batch_size
        self._weights = [0.005, 0.01, 0.02, 0.08, 0.32]
        self._num_iters = args.num_iters

        self.f1_score_bal_loss = f1_score_bal_loss
        self.occ_activ = nn.Sigmoid()

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]
            outputs_occ = [output_dict[key] for key in ["occ2", "occ3", "occ4", "occ5", "occ6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["target1"]
            target_b = self._args.div_flow * target_dict["target2"]
            target_occ_f = target_dict["target_occ1"]
            target_occ_b = target_dict["target_occ2"]

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    flow_loss = flow_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum()) * 0.5

            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ_f, output_ii[0][0])
                target_occ_b = _downsample2d_as(target_occ_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    output_occ_f = self.occ_activ(output_ii_jj[0])
                    output_occ_b = self.occ_activ(output_ii_jj[1])
                    bce_f_ii = self.f1_score_bal_loss(output_occ_f, target_occ_f)
                    bce_b_ii = self.f1_score_bal_loss(output_occ_b, target_occ_b)
                    occ_loss = occ_loss + self._weights[ii] * (bce_f_ii + bce_b_ii) * 0.5

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size / self._num_iters
            loss_dict["occ_loss"] = occ_loss / self._batch_size / self._num_iters
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / self._num_iters
        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow1"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ1"])))

        return loss_dict

class MultiScaleEPE_FlowNet_IRR_Bi_Occ_upsample(nn.Module):
    def __init__(self,
                 args):
        super(MultiScaleEPE_FlowNet_IRR_Bi_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.batch_size        
        self._weights = [0.0003125, 0.00125, 0.005, 0.01, 0.02, 0.08, 0.32]
        
        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = [output_dict[key] for key in ["flow", "flow1", "flow2", "flow3", "flow4", "flow5", "flow6"]]
            outputs_occ = [output_dict[key] for key in ["occ", "occ1", "occ2", "occ3", "occ4", "occ5", "occ6"]]

            # div_flow trick
            target_f = self._args.div_flow * target_dict["target1"]
            target_b = self._args.div_flow * target_dict["target2"]
            target_occ_f = target_dict["target_occ1"]
            target_occ_b = target_dict["target_occ2"]

            num_iters = len(outputs_flo[0])
            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                target_f_ii = _downsample2d_as(target_f, output_ii[0][0])
                target_b_ii = _downsample2d_as(target_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    epe_f_ii = _elementwise_epe(output_ii_jj[0], target_f_ii)
                    epe_b_ii = _elementwise_epe(output_ii_jj[1], target_b_ii)
                    flow_loss = flow_loss + self._weights[ii] * (epe_f_ii.sum() + epe_b_ii.sum()) * 0.5

            for ii, output_ii in enumerate(outputs_occ):
                target_occ_f = _downsample2d_as(target_occ_f, output_ii[0][0])
                target_occ_b = _downsample2d_as(target_occ_b, output_ii[0][1])
                for jj, output_ii_jj in enumerate(output_ii):
                    output_occ_f = self.occ_activ(output_ii_jj[0])
                    output_occ_b = self.occ_activ(output_ii_jj[1])
                    bce_f_ii = self.f1_score_bal_loss(output_occ_f, target_occ_f)
                    bce_b_ii = self.f1_score_bal_loss(output_occ_b, target_occ_b)
                    occ_loss = occ_loss + self._weights[ii] * (bce_f_ii + bce_b_ii) * 0.5

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size / num_iters
            loss_dict["occ_loss"] = occ_loss / self._batch_size / num_iters
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size / num_iters
        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict



class MultiScaleEPE_PWC(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target = self._args.div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] = total_loss / self._batch_size

        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_PWC_edge(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_edge, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target = self._args.div_flow * target_dict["target1"]

            total_loss = 0
            total_edge_loss = 0
            for ii, output_ii in enumerate(outputs):
                downsample = _downsample2d_as(target, output_ii)
                loss_ii = _elementwise_epe(output_ii, downsample).sum()
                loss_edge_ii = _elementwise_epe(output_ii[:,:,:-1,:-1]-output_ii[:,:,1:,1:], downsample[:,:,:-1,:-1]-downsample[:,:,1:,1:]).sum()
                total_loss = total_loss + self._weights[ii] * (loss_ii + loss_edge_ii)
                # total_loss = total_loss + self._weights[ii] * (loss_ii)
                total_edge_loss = total_edge_loss+ self._weights[ii] * (loss_edge_ii)
            loss_dict["total_loss"] = total_loss / self._batch_size
            loss_dict["total_edge_loss"] = total_edge_loss / self._batch_size

        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_PWC_Bi(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target_f = self._args.div_flow * target_dict["target1"]
            target_b = self._args.div_flow * target_dict["target2"]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                epe_i_f = _elementwise_epe(output_i[0], _downsample2d_as(target_f, output_i[0]))
                epe_i_b = _elementwise_epe(output_i[1], _downsample2d_as(target_b, output_i[1]))
                total_loss = total_loss + self._weights[i] * (epe_i_f.sum() + epe_i_b.sum())
            loss_dict["total_loss"] = total_loss / (2 * self._batch_size)
        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_PWC_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Occ, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo = self._args.div_flow * target_dict["target1"]
            target_occ = target_dict["target_occ1"]

            flow_loss = 0
            occ_loss = 0

            for i, output_i in enumerate(output_flo):
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i, _downsample2d_as(target_flo, output_i)).sum()

            for i, output_i in enumerate(output_occ):
                output_occ = self.occ_activ(output_i)
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ, _downsample2d_as(target_occ, output_occ))

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ, self).__init__()
        self._args = args
        self._batch_size = args.batch_size        
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["target1"]
            target_flo_b = self._args.div_flow * target_dict["target2"]
            target_occ_f = target_dict["target_occ1"]
            target_occ_b = target_dict["target_occ2"]

            # bchw
            flow_loss = 0
            occ_loss = 0

            for i, output_i in enumerate(output_flo):
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i[0], _downsample2d_as(target_flo_f, output_i[0])).sum()
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i[1], _downsample2d_as(target_flo_b, output_i[1])).sum()

            for i, output_i in enumerate(output_occ):
                output_occ_f = self.occ_activ(output_i[0])
                output_occ_b = self.occ_activ(output_i[1])
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / (2 * self._batch_size)
            loss_dict["occ_loss"] = occ_loss / (2 * self._batch_size) 
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / (2 * self._batch_size)

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ_upsample(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss


    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["target1"]
            target_flo_b = self._args.div_flow * target_dict["target2"]
            target_occ_f = target_dict["target_occ1"]
            target_occ_b = target_dict["target_occ2"]

            # bchw
            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj], _downsample2d_as(target_flo_f, output_ii[2 * jj])).sum()
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj + 1], _downsample2d_as(target_flo_b, output_ii[2 * jj + 1])).sum()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii)

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_occ_b = self.occ_activ(output_ii[2 * jj + 1])
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii)

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ_upsample_edge(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample_edge, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss


    def forward(self, output_dict, target_dict):
        loss_dict = {}
        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["target1"]
            target_flo_b = self._args.div_flow * target_dict["target2"]
            target_occ_f = target_dict["target_occ1"]
            target_occ_b = target_dict["target_occ2"]

            # bchw
            flow_loss = 0
            edge_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                loss_edge_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    downsample_f = _downsample2d_as(target_flo_f, output_ii[2 * jj])
                    downsample_b = _downsample2d_as(target_flo_b, output_ii[2 * jj + 1])
                    loss_ii      = loss_ii      + _elementwise_epe(output_ii[2 * jj],     downsample_f).sum()
                    loss_ii      = loss_ii      + _elementwise_epe(output_ii[2 * jj + 1], downsample_b).sum()
                    loss_edge_ii = loss_edge_ii + _elementwise_epe(output_ii[2 * jj][:,:,:-1,:-1]-output_ii[2 * jj][:,:,1:,1:], downsample_f[:,:,:-1,:-1]-downsample_f[:,:,1:,1:]).sum()
                    loss_edge_ii = loss_edge_ii + _elementwise_epe(output_ii[2 * jj + 1][:,:,:-1,:-1]-output_ii[2 * jj + 1][:,:,1:,1:], downsample_b[:,:,:-1,:-1]-downsample_b[:,:,1:,1:]).sum()
                edge_loss = edge_loss + self._weights[ii] * (loss_edge_ii) / len(output_ii)
                flow_loss = flow_loss + self._weights[ii] * (loss_ii) / len(output_ii)

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_occ_b = self.occ_activ(output_ii[2 * jj + 1])
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii)

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            e_loss = edge_loss.detach()

            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1
            
            if (e_loss.data > f_loss.data).numpy:
                e_l_w = 0
            else:
                e_l_w = f_loss / e_loss
            
            # f_l_w = 1
            # e_l_w = 1
            # o_l_w = 1


            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["edge_loss"] = edge_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + edge_loss * e_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.occ_loss_bce = nn.BCELoss(reduction='sum')

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["target1"]
            target_occ_f = target_dict["target_occ1"]

            # bchw
            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    loss_ii = loss_ii + _elementwise_robust_epe_char(output_ii[2 * jj], _downsample2d_as(target_flo_f, output_ii[2 * jj])).sum()
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                    loss_ii = loss_ii + self.occ_loss_bce(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel_edge(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample_Sintel_edge, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.occ_loss_bce = nn.BCELoss(reduction='sum')

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["target1"]
            target_occ_f = target_dict["target_occ1"]

            # bchw
            flow_loss = 0
            edge_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                loss_edge_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    downsample_f = _downsample2d_as(target_flo_f, output_ii[2 * jj])
                    loss_ii      = loss_ii      + _elementwise_robust_epe_char(output_ii[2 * jj], downsample_f).sum()
                    loss_edge_ii = loss_edge_ii + _elementwise_epe(output_ii[2 * jj][:,:,:-1,:-1]-output_ii[2 * jj][:,:,1:,1:], downsample_f[:,:,:-1,:-1]-downsample_f[:,:,1:,1:]).sum()
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                edge_loss = edge_loss + self._weights[ii] * (loss_edge_ii) / len(output_ii)
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                    loss_ii = loss_ii + self.occ_loss_bce(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["edge_loss"] = edge_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = ((flow_loss+edge_loss) * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample_KITTI, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.001, 0.001, 0.001, 0.002, 0.004, 0.004, 0.004]

        self.occ_activ = nn.Sigmoid()
        
    def forward(self, output_dict, target_dict):
        loss_dict = {}

        valid_mask = target_dict["input_valid"]
        b, _, h, w = target_dict["target1"].size()

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["target1"]

            # bchw
            flow_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii[2 * jj], target_flo_f), target_flo_f) * valid_mask

                    for bb in range(0, b):
                        valid_epe[bb, ...][valid_mask[bb, ...] == 0] = valid_epe[bb, ...][valid_mask[bb, ...] == 0].detach()
                        norm_const = h * w / (valid_mask[bb, ...].sum())
                        loss_ii = loss_ii + valid_epe[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const

                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii) * 2

            for ii, output_ii in enumerate(output_occ):
                for jj in range(0, len(output_ii) // 2):
                    output_ii[2 * jj] = output_ii[2 * jj].detach()
                    output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["total_loss"] = flow_loss / self._batch_size

        else:
            flow_gt_mag = torch.norm(target_dict["target1"], p=2, dim=1, keepdim=True) + 1e-8
            flow_epe = _elementwise_epe(output_dict["flow"], target_dict["target1"]) * valid_mask

            epe_per_image = (flow_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["epe"] = epe_per_image.mean()

            outlier_epe = (flow_epe > 3).float() * ((flow_epe / flow_gt_mag) > 0.05).float() * valid_mask
            outlier_per_image = (outlier_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["outlier"] = outlier_per_image.mean()

        return loss_dict

class MultiScaleEPE_PWC_KITTI(nn.Module):
    def __init__(self,
                 args):

        super(MultiScaleEPE_PWC_KITTI, self).__init__()
        self._args = args
        self._batch_size = args.batch_size_train
        self._weights = [0.001, 0.001, 0.001, 0.002, 0.004, 0.004, 0.004]
        
    def forward(self, output_dict, target_dict):
        loss_dict = {}

        valid_mask = target_dict["input_valid"]
        b, _, h, w = target_dict["target1"].size()

        if self.training:
            output_flo = output_dict['flow']

            # div_flow trick
            target_flo_f = self._args.div_flow * target_dict["target1"]

            # bchw
            flow_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                # for jj in range(0, len(output_ii) // 2):
                valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii, target_flo_f), target_flo_f) * valid_mask

                for bb in range(0, b):
                    # valid_epe[bb, ...][valid_mask[bb, ...] == 0] = valid_epe[bb, ...][valid_mask[bb, ...] == 0].detach()
                    norm_const = h * w / (valid_mask[bb, ...].sum())
                    loss_ii = loss_ii + valid_epe[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const

                # output_ii = output_ii.detach()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii)


            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["total_loss"] = flow_loss / self._batch_size
            # loss_dict["total_loss"] = valid_epe.mean()

        else:
            flow_gt_mag = torch.norm(target_dict["target1"], p=2, dim=1, keepdim=True) + 1e-8
            flow_epe = _elementwise_epe(output_dict["flow"], target_dict["target1"]) * valid_mask

            epe_per_image = (flow_epe.contiguous().view(b, -1).sum(1)) / (valid_mask.contiguous().view(b, -1).sum(1))
            loss_dict["epe"] = epe_per_image.mean()

            outlier_epe = (flow_epe > 3).float() * ((flow_epe / flow_gt_mag) > 0.05).float() * valid_mask
            outlier_per_image = (outlier_epe.contiguous().view(b, -1).sum(1)) / (valid_mask.contiguous().view(b, -1).sum(1))
            loss_dict["outlier"] = outlier_per_image.mean()

        return loss_dict

class sequence_loss(nn.Module):
    def __init__(self,
                args):

        super(sequence_loss, self).__init__()
        self._args = args        
        self._batch_size = args.batch_size_train
        self.MAX_FLOW = 1000

    def forward(self, output_dict, target_dict):
        """ Loss function defined over sequence of flow predictions """

        output_flo = output_dict['flow']
        target_flo_f = target_dict["target1"]
        loss_dict = {}

        if self.training:
            n_predictions = len(output_flo) 
            flow_loss = 0.0

            # exlude invalid pixels and extremely large diplacements
            valid_f = target_flo_f.abs().sum(dim=1) < self.MAX_FLOW

            for i in range(n_predictions):
                i_weight = 0.8**(n_predictions - i - 1)
                i_loss = (output_flo[i][0] - target_flo_f).abs()
                flow_loss += i_weight * (valid[:, None] * i_loss).mean()

            epe = torch.sum((output_flo[-1][0] - target_flo_f)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid_f.view(-1)]

            metrics = {
                'epe': epe.mean().item(),
                '1px': (epe < 1).float().mean().item(),
                '3px': (epe < 3).float().mean().item(),
                '5px': (epe < 5).float().mean().item(),
            }

            f_loss = flow_loss.detach()
            
            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["total_loss"] = flow_loss / self._batch_size

            return loss_dict, metrics

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][-1], target_dict["target1"]).mean()

            return loss_dict

class sequence_loss_occ(nn.Module):
    def __init__(self):

        super(sequence_loss_occ, self).__init__()  
        self.occ_activ = nn.Sigmoid()
        self.MAX_FLOW = 1000

        # self.occ_loss = f1_score_bal_loss
        self.occ_loss = nn.BCELoss(reduction='mean')
        self._batch_size = args.batch_size_train

    def forward(self, output_dict, target_dict):
        """ Loss function defined over sequence of flow predictions """

        output_flo = output_dict['flow']
        output_occ = output_dict['occ']
        target_flo_f = target_dict["target1"]
        target_flo_b = target_dict["target2"]
        target_occ_f = target_dict["target_occ1"]
        target_occ_b = target_dict["target_occ2"]

        loss_dict = {}
        
        if self.training:
            n_predictions = len(output_flo) 
            flow_loss = 0.0
            occ_loss = 0.0
            
            # exlude invalid pixels and extremely large diplacements
            valid_f = target_flo_f.abs().sum(dim=1) < self.MAX_FLOW
            valid_b = target_flo_b.abs().sum(dim=1) < self.MAX_FLOW

            for i in range(n_predictions):
                i_weight = 0.8**(n_predictions - i - 1)
                # i_loss = valid_f[:, None] * _elementwise_epe(output_flo[i][0], target_flo_f).sum() + valid_b[:, None] * _elementwise_epe(output_flo[i][1], target_flo_b).sum()
                i_loss = (output_flo[i][0] - target_flo_f).abs() + (output_flo[i][1] - target_flo_b).abs()
                flow_loss = flow_loss + i_weight * i_loss.mean()

                output_occ_f = self.occ_activ(output_occ[i][0])
                output_occ_b = self.occ_activ(output_occ[i][1])
                i_loss = self.occ_loss(output_occ_f, target_occ_f) + self.occ_loss(output_occ_b, target_occ_b)
                occ_loss = occ_loss + i_weight * i_loss.mean()

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()*5
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = (flow_loss / self._batch_size).detach()
            loss_dict["occ_loss"] = (occ_loss / self._batch_size).detach()
            loss_dict["total_loss"] = ((flow_loss) * f_l_w + occ_loss * o_l_w) / self._batch_size

            epe = torch.sum((output_flo[-1][0] - target_flo_f)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid_f.view(-1)]

            metrics = {
                'epe': epe.mean().item(),
                'f_loss':loss_dict["flow_loss"].item(),
                'o_loss':loss_dict["occ_loss"].item(),
                '1px': (epe < 1).float().mean().item(),
                '3px': (epe < 3).float().mean().item(),
                '5px': (epe < 5).float().mean().item(),
            }

            return loss_dict, metrics

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][0], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"][0])))

            return loss_dicts

class sequence_loss_occ_sintel(nn.Module):
    def __init__(self):

        super(sequence_loss_occ_sintel, self).__init__()  
        self.occ_activ = nn.Sigmoid()
        self.MAX_FLOW = 1000

        # self.occ_loss = f1_score_bal_loss
        self.occ_loss = nn.BCELoss(reduction='mean')
        

    def forward(self, output_dict, target_dict):
        """ Loss function defined over sequence of flow predictions """

        output_flo = output_dict['flow']
        output_occ = output_dict['occ']
        target_flo_f = target_dict["target1"]
        target_occ_f = target_dict["target_occ1"]

        loss_dict = {}
        
        if self.training:
            n_predictions = len(output_flo) 
            flow_loss = 0.0
            occ_loss = 0.0

            batch_size = output_flo.size()[0]
            
            # exlude invalid pixels and extremely large diplacements
            valid_f = target_flo_f.abs().sum(dim=1) < self.MAX_FLOW

            for i in range(n_predictions):
                i_weight = 0.8**(n_predictions - i - 1)
                # i_loss = valid_f[:, None] * _elementwise_epe(output_flo[i][0], target_flo_f).sum() + valid_b[:, None] * _elementwise_epe(output_flo[i][1], target_flo_b).sum()
                i_loss = (output_flo[i][0] - target_flo_f).abs()
                flow_loss = flow_loss + i_weight * i_loss.mean()

                output_occ_f = self.occ_activ(output_occ[i][0])
                i_loss = self.occ_loss(output_occ_f, target_occ_f)
                occ_loss = occ_loss + i_weight * i_loss.mean()

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()*5
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = (flow_loss / batch_size).detach()
            loss_dict["occ_loss"] = (occ_loss / batch_size).detach()
            loss_dict["total_loss"] = ((flow_loss) * f_l_w + occ_loss * o_l_w) / batch_size

            epe = torch.sum((output_flo[-1][0] - target_flo_f)**2, dim=1).sqrt()
            epe = epe.view(-1)[valid_f.view(-1)]

            metrics = {
                'epe': epe.mean().item(),
                'f_loss':loss_dict["flow_loss"].item(),
                'o_loss':loss_dict["occ_loss"].item(),
                '1px': (epe < 1).float().mean().item(),
                '3px': (epe < 3).float().mean().item(),
                '5px': (epe < 5).float().mean().item(),
            }

            return loss_dict, metrics

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][0], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"][0])))

            return loss_dict

def get_grid(x):
    grid_H = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    grid_V = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([grid_H, grid_V], 1)
    grids_cuda = grid.float().requires_grad_(False).type_as(x)
    return grids_cuda

class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, height_im, width_im, div_flow):
        flo_list = []
        flo_w = flow[:, 0] * 2 / max(width_im - 1, 1) / div_flow
        flo_h = flow[:, 1] * 2 / max(height_im - 1, 1) / div_flow
        flo_list.append(flo_w)
        flo_list.append(flo_h)
        flow_for_grid = torch.stack(flo_list).transpose(0, 1)
        grid = torch.add(get_grid(x), flow_for_grid).transpose(1, 2).transpose(2, 3)        
        x_warp = F.grid_sample(x, grid, align_corners=True)

        mask = torch.ones(x.size(), requires_grad=False).type_as(x)
        mask = F.grid_sample(mask, grid, align_corners=True)
        # mask = (mask >= 1.0).float()
        mask[mask<0.9999] = 0
        mask[mask>0] = 1

        return x_warp * mask

def Forward_Warp_Python(im0, flow, interpolation_mode=0):
    '''
    im0: the first image with shape [B, C, H, W]
    flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
    interpolation_mode: 0 is Bilinear, 1 is Nearest
    '''
    im1 = torch.zeros_like(im0)
    B = im0.shape[0]
    H = im0.shape[2]
    W = im0.shape[3]
    if interpolation_mode == 0:
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    x = w + flow[b, 0, h, w]
                    y = h + flow[b, 1, h, w]
                    nw = (int(torch.floor(x)), int(torch.floor(y)))
                    ne = (nw[0]+1, nw[1])
                    sw = (nw[0], nw[1]+1)
                    se = (nw[0]+1, nw[1]+1)
                    p = im0[b, :, h, w]
                    if nw[0] >= 0 and se[0] < W and nw[1] >= 0 and se[1] < H:
                        nw_k = (se[0]-x)*(se[1]-y)
                        ne_k = (x-sw[0])*(sw[1]-y)
                        sw_k = (ne[0]-x)*(y-ne[1])
                        se_k = (x-nw[0])*(y-nw[1])
                        im1[b, :, nw[1], nw[0]] += nw_k*p
                        im1[b, :, ne[1], ne[0]] += ne_k*p
                        im1[b, :, sw[1], sw[0]] += sw_k*p
                        im1[b, :, se[1], se[0]] += se_k*p
    else:
        round_flow = torch.round(flow)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    x = w + int(round_flow[b, h, w, 0])
                    y = h + int(round_flow[b, h, w, 1])
                    if x >= 0 and x < W and y >= 0 and y < H:
                        im1[b, :, y, x] = im0[b, :, h, w]
    return im1

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, args, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.warping_layer = WarpingLayer()

    # def forward(self, img1, img2):
    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']
            print("SSIM functions are not currently supported for training")
            return 0 
        else:
            _, channel, h, w = target_dict["input1"].size()

            if channel == self.channel and self.window.data.type() == target_dict["input1"].data.type():
                window = self.window
            else:
                window = create_window(self.window_size, channel)
                
                if target_dict["input1"].is_cuda:
                    window = window.cuda(target_dict["input1"].get_device())
                window = window.type_as(target_dict["input1"])
                
                self.window = window
                self.channel = channel
            
            x2_warp = self.warping_layer(target_dict["input2"], output_dict["flow"], h, w, 1)
            ssim = _ssim(x2_warp, target_dict["input1"], window, self.window_size, channel, self.size_average)

            loss_dict["ssim"] = 1 - ssim
            return loss_dict

class PSNR(torch.nn.Module):
    def __init__(self, args):
        super(PSNR, self).__init__()

        self.warping_layer = WarpingLayer()

    # def forward(self, img1, img2):
    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']
            print("PSNR functions are not currently supported for training")
            return 0 
        else:
            _, channel, h, w = target_dict["input1"].size()
            
            x2_warp = self.warping_layer(target_dict["input2"], output_dict["flow"], h, w, 1)
            # x2_warp = self.warping_layer(target_dict["input2"], target_dict["target1"], h, w, 1)

            mse = ((x2_warp - target_dict["input1"]) ** 2).mean()
            if mse == 0:
                return 100
            PIXEL_MAX = 1.0

            loss_dict["psnr"] = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    
            return loss_dict



class Sharpness_diff(torch.nn.Module):
    #这玩意只能运行在gpu下？？？？
    def __init__(self, args):
        super(Sharpness_diff, self).__init__()
        
        self.args = args
        self.warping_layer = WarpingLayer()

    def log10(self,tf, t):
        """
        Calculates the base-10 log of each element in t.
        @param t: The tensor from which to calculate the base-10 log.
        @return: A tensor with the base-10 log of each element in t.
        """
        numerator = tf.log(t)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def sharp_diff_error(self,tf, gen_frames, gt_frames):
        """
        Computes the Sharpness Difference error between the generated images and the ground truth
        images.
        @param gen_frames: A tensor of shape [batch_size, height, width, 3]. The frames generated by the
                        generator model.
        @param gt_frames: A tensor of shape [batch_size, height, width, 3]. The ground-truth frames for
                        each frame in gen_frames.
        @return: A scalar tensor. The Sharpness Difference error over each frame in the batch.
        """
        shape = tf.shape(gen_frames)
        num_pixels = tf.to_float(shape[1] * shape[2] * shape[3])

        # gradient difference
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        # TODO: Could this be simplified with one filter [[-1, 2], [0, -1]]?
        pos = tf.constant(np.identity(3), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        print(filter_x)
        print(filter_y)
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_frames, filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames, filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_frames, filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_frames, filter_y, strides, padding=padding))

        gen_grad_sum = gen_dx + gen_dy
        gt_grad_sum = gt_dx + gt_dy

        grad_diff = tf.abs(gt_grad_sum - gen_grad_sum)

        batch_errors = 10 * self.log10(tf,1 / ((1 / num_pixels) * tf.reduce_sum(grad_diff, [1, 2, 3])))
        # return tf.reduce_mean(batch_errors)
        return batch_errors
        

    def forward(self, output_dict, target_dict):
        
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']
            print("Sharpness_diff functions are not currently supported for training")
            return 0 
        else:
            import tensorflow as tf

            _, channel, h, w = target_dict["input1"].size()
            num_pixels = channel*h*w

            predict_frame = tf.compat.v1.placeholder(tf.float32,shape=[None,h,w,channel])
            grouth_truth  = tf.compat.v1.placeholder(tf.float32,shape=[None,h,w,channel])
            sharpError = self.sharp_diff_error(tf,predict_frame ,grouth_truth)
            
            x2_warp = self.warping_layer(target_dict["input2"], output_dict["flow"], h, w, 1)
            # x2_warp = target_dict["input2"]

            predictImageour = tensor2numpy(x2_warp)
            groundTruthour = tensor2numpy(target_dict["input1"])

            with tf.Session() as sess:
                sEour1 = sess.run(sharpError, feed_dict={predict_frame: predictImageour, grouth_truth: groundTruthour})
            
            loss_dict["sharpness"] = sEour1
            return loss_dict


