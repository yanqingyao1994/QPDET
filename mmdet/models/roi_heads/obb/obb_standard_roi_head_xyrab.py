import torch

from mmdet.core import arb2roi
from mmdet.core import arb2result
from mmdet.models.builder import HEADS
from .obb_standard_roi_head import OBBStandardRoIHead

@HEADS.register_module()
class OBBStandardRoIHeadXYRAB(OBBStandardRoIHead):
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        for proposal in proposals:
            proposal[:, 5:] = torch.cat((proposal[:, 6:], proposal[:, 5:6]), -1)
        rois = arb2roi(proposals, bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois[:, :6])
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None):
        # assign gts and sample proposals
        if self.with_bbox:
            start_bbox_type = self.bbox_head.start_bbox_type
            end_bbox_type = self.bbox_head.end_bbox_type
            target_bboxes = gt_bboxes if start_bbox_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore \
                    if start_bbox_type == 'hbb' else gt_obboxes_ignore

            num_imgs = len(img_metas)
            if target_bboxes_ignore is None:
                target_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i][:, :6], target_bboxes[i],
                    target_bboxes_ignore[i], gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    target_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if start_bbox_type != end_bbox_type:
                    if gt_obboxes[i].numel() == 0:
                        sampling_result.pos_gt_bboxes = gt_obboxes[i].new_zeros(
                            (0, gt_obboxes[0].size(-1)))
                    else:
                        sampling_result.pos_gt_bboxes = \
                                gt_obboxes[i][sampling_result.pos_assigned_gt_inds, :]

                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training"""
        rois = arb2roi([res.bboxes for res in sampling_results],
                       bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois[:, :6])

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
