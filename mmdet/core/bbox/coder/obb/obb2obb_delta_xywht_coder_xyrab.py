import numpy as np
import torch

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import  obb2poly, rectpoly2obb
# from mmdet.core.bbox.transforms_obb import regular_theta, regular_obb
from mmdet.core.bbox.builder import BBOX_CODERS

pi = 3.1415926


@BBOX_CODERS.register_module()
class OBB2OBBDeltaXYWHTCoderXYRAB(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        encoded_bboxes = obb2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16/1000):
        decoded_bboxes = delta2obb(bboxes, pred_bboxes, self.means, self.stds,
                                   wh_ratio_clip)

        return decoded_bboxes


def obb2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    px, py, pr, pa, pb = proposals.float()[..., 6:].unbind(-1)

    poly_g = obb2poly(gt.float())
    gx = gt[..., 0] 
    gy = gt[..., 1] 
    gw = gt[..., 2]
    gh = gt[..., 3]
    gr = torch.sqrt(torch.square(gw * 0.5) + torch.square(gh *0.5))

    xg_coor, yg_coor = poly_g[:, 0::2], poly_g[:, 1::2]
    yg_min, _ = torch.min(yg_coor, dim=1, keepdim=True)
    xg_max, _ = torch.max(xg_coor, dim=1, keepdim=True)

    _xg_coor = xg_coor.clone()
    _xg_coor[torch.abs(yg_coor-yg_min) > 0.1] = -1000
    ga, _ = torch.max(_xg_coor, dim=1)

    _yg_coor = yg_coor.clone()
    _yg_coor[torch.abs(xg_coor-xg_max) > 0.1] = -1000
    gb, _ = torch.max(_yg_coor, dim=1)

    dx = (gx - px) / (pr * 2)
    dy = (gy - py) / (pr * 2)
    dr = torch.log(gr / pr)
    da = torch.asin((ga - gx) / gr) - pa
    db = torch.asin((gb - gy) / gr) - pb

    da = da.clamp(min=-pi/2, max=pi/2)
    db = db.clamp(min=-pi/2, max=pi/2)

    deltas = torch.stack([dx, dy, dr, da, db], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2obb(proposals,
              deltas,
              means=(0., 0., 0., 0., 0.),
              stds=(1., 1., 1., 1., 1.),
              wh_ratio_clip=16/1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx, dy, dr, da, db = denorm_deltas.unbind(dim=-1)
    px, py, pr, pa, pb = proposals[..., 5:].unbind(dim=-1)

    gx = px + 2 * pr * dx
    gy = py + 2 * pr * dy
    gr = pr * dr.exp()
    ga = pa + da
    gb = pb + db

    da = da.clamp(min=-pi/2, max=pi/2)
    db = db.clamp(min=-pi/2, max=pi/2)

    x1 = gx + gr * torch.sin(ga)
    y1 = gy - gr * torch.cos(ga)
    x2 = gx + gr * torch.cos(gb)
    y2 = gy + gr * torch.sin(gb)
    x3 = gx - gr * torch.sin(ga)
    y3 = gy + gr * torch.cos(ga)
    x4 = gx - gr * torch.cos(gb)
    y4 = gy - gr * torch.sin(gb)

    polys = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)
    bboxes = rectpoly2obb(polys)
    return bboxes
