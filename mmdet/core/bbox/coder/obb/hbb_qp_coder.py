import torch

from ..base_bbox_coder import BaseBBoxCoder
from mmdet.core.bbox.transforms_obb import obb2poly, rectpoly2obb
from mmdet.core.bbox.builder import BBOX_CODERS

@BBOX_CODERS.register_module()
class HBBQPCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        encoded_bboxes = bbox2delta_sp(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta_sp2bbox(bboxes, pred_bboxes, self.means,
                                       self.stds, wh_ratio_clip)
        return decoded_bboxes

calc = []

def bbox2delta_sp(proposals, gt,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.)):
    assert proposals.size(0) == gt.size(0)
    # proposals为x0,y0,x1,y1, gt为x,y,w,h,theta
    proposals = proposals.float()
    gt = gt.float()

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    pr = torch.sqrt(torch.square(pw * 0.5) + torch.square(ph * 0.5))

    poly = obb2poly(gt)
    gx = gt[..., 0]
    gy = gt[..., 1]
    gw = gt[..., 2]
    gh = gt[..., 3]
    gr = torch.sqrt(torch.square(gw * 0.5) + torch.square(gh * 0.5))

    x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)
    #计算y对应的x的坐标值
    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor-y_min) > 0.1] = -1000
    ga, _ = torch.max(_x_coor, dim=1)
    #计算x对应的y的坐标值
    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor-x_max) > 0.1] = -1000
    gb, _ = torch.max(_y_coor, dim=1)

    dx = (gx - px) / (pr * 2)
    dy = (gy - py) / (pr * 2)
    dr = torch.log(gr / pr)
    da = torch.asin((ga - gx) / gr)
    db = torch.asin((gb - gy) / gr)
    deltas = torch.stack([dx, dy, dr, da, db], dim=-1)

    global calc
    calc = calc[-9999:]
    calc.append(deltas.cpu().detach())
    m = torch.cat(calc).mean(0)
    s = torch.cat(calc).std(0)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta_sp2bbox(proposals, deltas,
                  means=(0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    pr = torch.sqrt(torch.square(pw * 0.5) + torch.square(ph * 0.5))

    dx, dy, dr, da, db = denorm_deltas.unbind(dim=-1)
    gx = px + 2 * pr * dx
    gy = py + 2 * pr * dy
    gr = pr * dr.exp()

    x1 = gx + gr * torch.sin(da)
    y1 = gy - gr * torch.cos(da)
    x2 = gx + gr * torch.cos(db)
    y2 = gy + gr * torch.sin(db)
    x3 = gx - gr * torch.sin(da)
    y3 = gy + gr * torch.cos(da)
    x4 = gx - gr * torch.cos(db)
    y4 = gy - gr * torch.sin(db)

    polys = torch.stack([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)
    obboxes = rectpoly2obb(polys)
    return obboxes
