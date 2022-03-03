from __future__ import absolute_import, division, print_function

import copy

import torch
import torch.nn as nn

import models.correlation as correlation
from models.i_pmca.pwc_modules import (ContextNetwork, FeatureExtractor,
                                       FlowEstimatorDense, OccContextNetwork,
                                       OccEstimatorDense, WarpingLayer, conv,
                                       initialize_msra, rescale_flow,
                                       upsample2d_as)

from models.i_pmca.irr_modules import OccUpsampleNetwork, RefineFlow, RefineOcc


class PMCA(nn.Module):
    def __init__(self,in_c):
        super(PMCA, self).__init__()

        in_c = in_c

        self.b0_conv1 = conv(in_c,  128)

        self.b1_conv1 = conv(128, 128, dilation=1)
        self.b1_conv2 = conv(128, 64, dilation=2)
        self.b1_conv3 = conv(64,  32, dilation=1)

        self.b2_conv1 = conv(128, 64, dilation=2)
        self.b2_conv2 = conv(128, 64, dilation=4)
        self.b2_conv3 = conv(64,  32, dilation=1)

        self.b3_conv1 = conv(128, 64, dilation=4)
        self.b3_conv2 = conv(128, 64, dilation=8)
        self.b3_conv3 = conv(64,  32, dilation=1)

        self.b4_conv1 = conv(128, 64, dilation=8)
        self.b4_conv2 = conv(128, 64, dilation=16)
        self.b4_conv3 = conv(64,  32, dilation=1)

        self.dc_conv1 = conv(32*4, 96, dilation=1)
        self.dc_conv2 = conv(96,2, isReLU=False)

    def forward(self, x):
        b0_out = self.b0_conv1(x)

        b1_i = self.b1_conv2(self.b1_conv1(b0_out))
        b1_out = self.b1_conv3(b1_i)
        b2_i   = self.b2_conv2(torch.cat((self.b2_conv1(b0_out), b1_i),1))
        b2_out = self.b2_conv3(b2_i)
        b3_i   = self.b3_conv2(torch.cat((self.b3_conv1(b0_out), b2_i),1))
        b3_out = self.b3_conv3(b3_i)
        b4_i   = self.b4_conv2(torch.cat((self.b4_conv1(b0_out), b3_i),1))
        b4_out = self.b4_conv3(b4_i)

        dcdcat = torch.cat((b1_out, b2_out, b3_out, b4_out),1)
        return self.dc_conv2(self.dc_conv1(dcdcat))

class OccPMCA(nn.Module):
    def __init__(self,in_c):
        super(OccPMCA, self).__init__()

        in_c = in_c

        self.b0_conv1 = conv(in_c,  128)

        self.b1_conv1 = conv(128, 128, dilation=1)
        self.b1_conv2 = conv(128, 64, dilation=2)
        self.b1_conv3 = conv(64,  32, dilation=1)

        self.b2_conv1 = conv(128, 64, dilation=2)
        self.b2_conv2 = conv(128, 64, dilation=4)
        self.b2_conv3 = conv(64,  32, dilation=1)

        self.b3_conv1 = conv(128, 64, dilation=4)
        self.b3_conv2 = conv(128, 64, dilation=8)
        self.b3_conv3 = conv(64,  32, dilation=1)

        self.b4_conv1 = conv(128, 64, dilation=8)
        self.b4_conv2 = conv(128, 64, dilation=16)
        self.b4_conv3 = conv(64,  32, dilation=1)

        self.dc_conv1 = conv(32*4, 96, dilation=1)
        self.dc_conv2 = conv(96,1, isReLU=False)

    def forward(self, x):
        b0_out = self.b0_conv1(x)

        b1_i = self.b1_conv2(self.b1_conv1(b0_out))
        b1_out = self.b1_conv3(b1_i)
        b2_i   = self.b2_conv2(torch.cat((self.b2_conv1(b0_out), b1_i),1))
        b2_out = self.b2_conv3(b2_i)
        b3_i   = self.b3_conv2(torch.cat((self.b3_conv1(b0_out), b2_i),1))
        b3_out = self.b3_conv3(b3_i)
        b4_i   = self.b4_conv2(torch.cat((self.b4_conv1(b0_out), b3_i),1))
        b4_out = self.b4_conv3(b4_i)

        dcdcat = torch.cat((b1_out, b2_out, b3_out, b4_out),1)
        return self.dc_conv2(self.dc_conv1(dcdcat))


class i_pmca(nn.Module):
    def __init__(self, args, PP_type='original'):# substitution parallel 
        super(i_pmca, self).__init__()
        assert(PP_type in ['original', 'substitution', 'parallel'])
        self.args = args
        self.type = PP_type
        self._div_flow = args.div_flow
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in_flo = self.dim_corr + 32 + 2
        self.num_ch_in_occ = self.dim_corr + 32 + 1

        self.flow_estimators = FlowEstimatorDense(self.num_ch_in_flo)
        self.occ_estimators = OccEstimatorDense(self.num_ch_in_occ)
        self.occ_shuffle_upsample = OccUpsampleNetwork(11, 1)

        if self.type == 'original':
            self.context_networks = ContextNetwork(self.num_ch_in_flo + 448 + 2)
            self.occ_context_networks = OccContextNetwork(self.num_ch_in_occ + 448 + 1)
        elif self.type == 'parallel':
            self.context_networks = ContextNetwork(self.num_ch_in_flo + 448 + 2)
            self.occ_context_networks = OccContextNetwork(self.num_ch_in_occ + 448 + 1)
            # self.context_networks_2 = PMCA(self.num_ch_in_flo + 448 + 2)
            self.occ_context_networks_2 =  OccPMCA(self.num_ch_in_occ + 448 + 1)
        elif self.type == 'substitution':
            # self.context_networks_2 = PMCA(self.num_ch_in_flo + 448 + 2)
            self.context_networks = ContextNetwork(self.num_ch_in_flo + 448 + 2)
            self.occ_context_networks_2 =  OccPMCA(self.num_ch_in_occ + 448 + 1)

        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1)])

        self.conv_1x1_1 = conv(16, 3, kernel_size=1, stride=1, dilation=1)

        self.refine_flow = RefineFlow(2 + 1 + 32)
        self.refine_occ = RefineOcc(1 + 32 + 32)

        initialize_msra(self.modules())

    def forward(self, input_dict):

        x1_raw = input_dict['input1']
        x2_raw = input_dict['input2']
        batch_size, _, height_im, width_im = x1_raw.size()

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        output_dict = {}
        output_dict_eval = {}
        flows = []
        occs = []

        _, _, h_x1, w_x1, = x1_pyramid[0].size()
        flow_f = torch.zeros(batch_size, 2, h_x1, w_x1).float().cuda()
        flow_b = torch.zeros(batch_size, 2, h_x1, w_x1).float().cuda()
        occ_f = torch.zeros(batch_size, 1, h_x1, w_x1).float().cuda()
        occ_b = torch.zeros(batch_size, 1, h_x1, w_x1).float().cuda()

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            x1 = x1.contiguous()
            x2 = x2.contiguous()
            if l <= self.output_level:

                # warping
                if l == 0:
                    x2_warp = x2
                    x1_warp = x1
                else:
                    flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                    flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                    occ_f = upsample2d_as(occ_f, x1, mode="bilinear")
                    occ_b = upsample2d_as(occ_b, x2, mode="bilinear")
                    x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                    x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)

                # correlation
                # out_corr_f = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x1, x2_warp)
                # out_corr_b = Correlation(pad_size=self.search_range, kernel_size=1, max_displacement=self.search_range, stride1=1, stride2=1, corr_multiply=1)(x2, x1_warp)
                out_corr_f = correlation.FunctionCorrelation(x1, x2_warp)
                out_corr_b = correlation.FunctionCorrelation(x2, x1_warp)
                out_corr_relu_f = self.leakyRELU(out_corr_f)
                out_corr_relu_b = self.leakyRELU(out_corr_b)

                if l != self.output_level:
                    x1_1by1 = self.conv_1x1[l](x1)
                    x2_1by1 = self.conv_1x1[l](x2)
                else:
                    x1_1by1 = x1
                    x2_1by1 = x2

                # concat and estimate flow
                flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=True)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=True)

                x_intm_f, flow_res_f = self.flow_estimators(torch.cat([out_corr_relu_f, x1_1by1, flow_f], dim=1))
                x_intm_b, flow_res_b = self.flow_estimators(torch.cat([out_corr_relu_b, x2_1by1, flow_b], dim=1))
                flow_est_f = flow_f + flow_res_f
                flow_est_b = flow_b + flow_res_b

                flow_cont_f = flow_est_f + self.context_networks(torch.cat([x_intm_f, flow_est_f], dim=1))
                flow_cont_b = flow_est_b + self.context_networks(torch.cat([x_intm_b, flow_est_b], dim=1))

                # occ estimation
                x_intm_occ_f, occ_res_f = self.occ_estimators(torch.cat([out_corr_relu_f, x1_1by1, occ_f], dim=1))
                x_intm_occ_b, occ_res_b = self.occ_estimators(torch.cat([out_corr_relu_b, x2_1by1, occ_b], dim=1))
                occ_est_f = occ_f + occ_res_f
                occ_est_b = occ_b + occ_res_b

                if self.type == 'original':
                    occ_cont_f = occ_est_f + self.occ_context_networks(torch.cat([x_intm_occ_f, occ_est_f], dim=1))
                    occ_cont_b = occ_est_b + self.occ_context_networks(torch.cat([x_intm_occ_b, occ_est_b], dim=1))
                elif self.type == 'parallel':
                    hook_ori = occ_est_f
                    hook_context = self.occ_context_networks(torch.cat([x_intm_occ_f, occ_est_f], dim=1))
                    hook_pmc = self.occ_context_networks_2(torch.cat([x_intm_occ_f, occ_est_f], dim=1))
                    occ_cont_f = occ_est_f + 0.5*self.occ_context_networks(torch.cat([x_intm_occ_f, occ_est_f], dim=1)) + 0.5*self.occ_context_networks_2(torch.cat([x_intm_occ_f, occ_est_f], dim=1))
                    occ_cont_b = occ_est_b + 0.5*self.occ_context_networks(torch.cat([x_intm_occ_b, occ_est_b], dim=1)) + 0.5*self.occ_context_networks_2(torch.cat([x_intm_occ_b, occ_est_b], dim=1))
                elif self.type == 'substitution':
                    occ_cont_f = occ_est_f + self.occ_context_networks_2(torch.cat([x_intm_occ_f, occ_est_f], dim=1))
                    occ_cont_b = occ_est_b + self.occ_context_networks_2(torch.cat([x_intm_occ_b, occ_est_b], dim=1))
                    hook_ori = occ_est_f
                    hook_pmc = self.occ_context_networks_2(torch.cat([x_intm_occ_f, occ_est_f], dim=1))

                # refinement
                img1_resize = upsample2d_as(x1_raw, flow_f, mode="bilinear")
                img2_resize = upsample2d_as(x2_raw, flow_b, mode="bilinear")
                img2_warp = self.warping_layer(img2_resize, rescale_flow(flow_cont_f, self._div_flow, width_im, height_im, to_local=False), height_im, width_im, self._div_flow)
                img1_warp = self.warping_layer(img1_resize, rescale_flow(flow_cont_b, self._div_flow, width_im, height_im, to_local=False), height_im, width_im, self._div_flow)

                # flow refine
                flow_f = self.refine_flow(flow_cont_f.detach(), img1_resize - img2_warp, x1_1by1)
                flow_b = self.refine_flow(flow_cont_b.detach(), img2_resize - img1_warp, x2_1by1)

                flow_cont_f = rescale_flow(flow_cont_f, self._div_flow, width_im, height_im, to_local=False)
                flow_cont_b = rescale_flow(flow_cont_b, self._div_flow, width_im, height_im, to_local=False)
                flow_f = rescale_flow(flow_f, self._div_flow, width_im, height_im, to_local=False)
                flow_b = rescale_flow(flow_b, self._div_flow, width_im, height_im, to_local=False)

                # occ refine
                x2_1by1_warp = self.warping_layer(x2_1by1, flow_f, height_im, width_im, self._div_flow)
                x1_1by1_warp = self.warping_layer(x1_1by1, flow_b, height_im, width_im, self._div_flow)

                occ_f = self.refine_occ(occ_cont_f.detach(), x1_1by1, x1_1by1 - x2_1by1_warp)
                occ_b = self.refine_occ(occ_cont_b.detach(), x2_1by1, x2_1by1 - x1_1by1_warp)

                flows.append([flow_cont_f, flow_cont_b, flow_f, flow_b])
                occs.append([occ_cont_f, occ_cont_b, occ_f, occ_b])

            else:
                flow_f = upsample2d_as(flow_f, x1, mode="bilinear")
                flow_b = upsample2d_as(flow_b, x2, mode="bilinear")
                flows.append([flow_f, flow_b])

                x2_warp = self.warping_layer(x2, flow_f, height_im, width_im, self._div_flow)
                x1_warp = self.warping_layer(x1, flow_b, height_im, width_im, self._div_flow)
                flow_b_warp = self.warping_layer(flow_b, flow_f, height_im, width_im, self._div_flow)
                flow_f_warp = self.warping_layer(flow_f, flow_b, height_im, width_im, self._div_flow)

                if l != self.num_levels-1:
                    x1_in = self.conv_1x1_1(x1)
                    x2_in = self.conv_1x1_1(x2)
                    x1_w_in = self.conv_1x1_1(x1_warp)
                    x2_w_in = self.conv_1x1_1(x2_warp)
                else:
                    x1_in = x1
                    x2_in = x2
                    x1_w_in = x1_warp
                    x2_w_in = x2_warp

                occ_f = self.occ_shuffle_upsample(occ_f, torch.cat([x1_in, x2_w_in, flow_f, flow_b_warp], dim=1))
                occ_b = self.occ_shuffle_upsample(occ_b, torch.cat([x2_in, x1_w_in, flow_b, flow_f_warp], dim=1))

                occs.append([occ_f, occ_b])

        output_dict_eval['flow'] = upsample2d_as(flow_f, x1_raw, mode="bilinear") * (1.0 / self._div_flow)
        output_dict_eval['occ'] = upsample2d_as(occ_f, x1_raw, mode="bilinear")
        output_dict['flow'] = flows
        output_dict['occ'] = occs

        output_dict_eval['hook_ori'] = hook_ori
        output_dict_eval['hook_context'] = hook_context
        output_dict_eval['hook_pmc'] = hook_pmc

        if self.training:
            return output_dict
        else:
            return output_dict_eval


class i_pmca_o(i_pmca):
    def __init__(self,
                 args):
        super(i_pmca_o, self).__init__(
            args,
            PP_type="original")

class i_pmca_s(i_pmca):
    def __init__(self,
                 args):
        super(i_pmca_s, self).__init__(
            args,
            PP_type="substitution")

class i_pmca_p(i_pmca):
    def __init__(self,
                 args):
        super(i_pmca_p, self).__init__(
            args,
            PP_type="parallel")