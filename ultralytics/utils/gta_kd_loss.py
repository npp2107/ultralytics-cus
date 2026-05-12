from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation. <https://arxiv.org/abs/2011.13256>`_.
    Modified for Ground Truth Aware (GTA) distillation.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, y_s, y_t, masks=None):
        """Forward computation.

        Args:
            y_s (list): The student model prediction with shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with shape (N, C, H, W) in list.
            masks (list, optional): List of Ground Truth masks with shape (N, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, _H, _W = s.shape

            s_flat = s.view(N, C, -1)
            t_flat = t.view(N, C, -1)

            if masks is not None and idx < len(masks):
                mask = masks[idx].view(N, 1, -1)  # (N, 1, HW)

                # Mask out background by setting to -1e9 before softmax to ignore those regions
                s_flat_m = s_flat.clone()
                t_flat_m = t_flat.clone()

                # Check for empty masks
                has_pixel = (mask.sum(dim=2) > 0).float()  # (N, 1)

                # Apply mask: set non-foreground pixels to a very low value
                s_flat_m[mask.expand_as(s_flat_m) == 0] = -1e9
                t_flat_m[mask.expand_as(t_flat_m) == 0] = -1e9

                softmax_pred_T = F.softmax(t_flat_m / self.tau, dim=2)
                logsoftmax = torch.nn.LogSoftmax(dim=2)

                cost = (
                    softmax_pred_T * logsoftmax(t_flat_m / self.tau) - softmax_pred_T * logsoftmax(s_flat_m / self.tau)
                ) * (self.tau**2)

                # Average over channels and batch for those images that have GT pixels
                channel_cost = (cost.sum(dim=2) * has_pixel).sum(dim=1) / (C + 1e-6)  # (N,)
                batch_cost = channel_cost.sum() / (has_pixel.sum() + 1e-6)
                losses.append(batch_cost)
            else:
                # Standard CWD
                softmax_pred_T = F.softmax(t_flat / self.tau, dim=2)
                logsoftmax = torch.nn.LogSoftmax(dim=2)
                cost = torch.sum(
                    softmax_pred_T * logsoftmax(t_flat / self.tau) - softmax_pred_T * logsoftmax(s_flat / self.tau)
                ) * (self.tau**2)
                losses.append(cost / (C * N))

        loss = sum(losses)
        return loss


class MGDLoss(nn.Module):
    """Masked Generative Distillation (MGD) loss with Ground Truth Awareness."""

    def __init__(
        self,
        student_channels,
        teacher_channels,
        alpha_mgd=0.00002,
        lambda_mgd=0.65,
    ):
        super().__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.generation = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                ).to(device)
                for channel in teacher_channels
            ]
        )

    def forward(self, y_s, y_t, masks=None, layer=None):
        """Forward computation.

        Args:
            y_s (list): The student model prediction.
            y_t (list): The teacher model prediction.
            masks (list, optional): Ground Truth regions masks.
        """
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if layer == "outlayer":
                idx = -1
            m = masks[idx] if masks is not None and idx < len(masks) else None
            losses.append(self.get_dis_loss(s, t, idx, mask=m) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx, mask=None):
        N, _C, H, W = preds_T.shape
        device = preds_S.device

        # MGD random masking for generation task
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        if mask is not None:
            # Distill only on Ground Truth Regions (GTA)
            # mask is (N, 1, H, W)
            loss = (new_fea - preds_T) ** 2 * mask
            dis_loss = loss.sum() / N
        else:
            loss_mse = nn.MSELoss(reduction="sum")
            dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller="mgd", loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.align_module = nn.ModuleList()
        self.norm1 = nn.ModuleList()

        for s_chan, t_chan in zip(channels_s, channels_t):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(t_chan, affine=False)
            ).to(device)
            self.align_module.append(align)

        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))

        if distiller == "mgd":
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == "cwd":
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t, masks=None):
        if len(y_s) != len(y_t):
            y_t = y_t[len(y_t) // 2 :]

        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # Match input dtype to module dtype
            s = s.type(next(self.align_module[idx].parameters()).dtype)
            t = t.type(next(self.align_module[idx].parameters()).dtype)

            if self.distiller == "cwd":
                s = self.align_module[idx](s)
                stu_feats.append(s)
                tea_feats.append(t.detach())
            else:
                t = self.norm1[idx](t)
                stu_feats.append(s)
                tea_feats.append(t.detach())

        loss = self.feature_loss(stu_feats, tea_feats, masks=masks)
        return self.loss_weight * loss


class GTADistillationLoss:
    """Ground Truth Aware Distillation Loss. Filters distillation to focus on regions corresponding to Ground Truth
    boxes.
    """

    def __init__(
        self,
        models,
        modelt,
        distiller="CWDLoss",
        distill_loss_weight=0.3,
        s_layers=["6", "8", "13", "16", "19", "22"],
        t_layers=["6", "8", "13", "16", "19", "22"],
        class_mapping=None,
        teacher_pred_conf=0.01,
    ):
        self.distiller = distiller
        self.s_layers = s_layers
        self.t_layers = t_layers
        self.models = models
        self.modelt = modelt
        self.distill_loss_weight = distill_loss_weight
        self.class_mapping = class_mapping
        self.teacher_pred_conf = teacher_pred_conf

        device = next(models.parameters()).device
        # Init warm up
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 192, 192).to(device)
            _ = self.models(dummy_input)
            _ = self.modelt(dummy_input)

        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []

        self._find_layers()

        self.distill_loss_fn = FeatureLoss(
            channels_s=self.channels_s,
            channels_t=self.channels_t,
            distiller=distiller[:3].lower(),
        )

    def _find_layers(self):
        for name, ml in self.modelt.named_modules():
            if name and name.startswith("model."):
                parts = name.split(".")
                if len(parts) >= 3 and parts[1] in self.t_layers and "cv2" in parts[2]:
                    if hasattr(ml, "conv"):
                        self.channels_t.append(ml.conv.out_channels)
                        self.teacher_module_pairs.append(ml)

        for name, ml in self.models.named_modules():
            if name and name.startswith("model."):
                parts = name.split(".")
                if len(parts) >= 3 and parts[1] in self.s_layers and "cv2" in parts[2]:
                    if hasattr(ml, "conv"):
                        self.channels_s.append(ml.conv.out_channels)
                        self.student_module_pairs.append(ml)

        nl = min(len(self.channels_s), len(self.channels_t))
        self.channels_s = self.channels_s[-nl:]
        self.channels_t = self.channels_t[-nl:]
        self.teacher_module_pairs = self.teacher_module_pairs[-nl:]
        self.student_module_pairs = self.student_module_pairs[-nl:]

    def register_hook(self):
        self.remove_handle_()
        self.teacher_outputs = []
        self.student_outputs = []

        def make_student_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    l.append(output.clone())
                else:
                    l.append([o.clone() if isinstance(o, torch.Tensor) else o for o in output])

            return forward_hook

        def make_teacher_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    l.append(output.detach().clone())
                else:
                    l.append([o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output])

            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_hook(self.student_outputs)))

    def get_loss(self, batch=None, teacher_preds=None):
        """Calculate distillation loss.

        Args:
            batch (dict, optional): Training batch containing 'bboxes' and 'batch_idx'.
            teacher_preds (tensor, optional): Teacher model predictions for the batch.
        """
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)

        if len(self.teacher_outputs) != len(self.student_outputs):
            print(
                f"Warning: Mismatched outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.student_outputs)}"
            )
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)

        # Handle teacher_preds if it's a tuple/list (standard YOLOv8 return)
        if isinstance(teacher_preds, (list, tuple)):
            teacher_preds = teacher_preds[0]

        masks = None
        if batch is not None:
            masks = self._generate_masks(batch, teacher_preds)

        quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs, masks=masks)

        if self.distiller.lower() != "cwdloss":
            quant_loss *= self.distill_loss_weight

        self.teacher_outputs.clear()
        self.student_outputs.clear()

        return quant_loss

    def _generate_masks(self, batch, teacher_preds=None):
        """Creates binary masks. Excludes regions where teacher model detects incorrectly (False Positives and False
        Negatives). Distills the remaining regions where teacher is likely correct (True Positives
        and Background).
        """
        bboxes_gt = batch.get("bboxes")
        batch_idx_gt = batch.get("batch_idx")
        cls_gt = batch.get("cls")
        if bboxes_gt is None or batch_idx_gt is None:
            return None

        # Pre-calculate errors per image in batch
        N = batch["img"].shape[0]
        img_h, img_w = batch["img"].shape[2:]
        device = bboxes_gt.device

        batch_errors = []  # List of list of (bboxes, value) for each image

        if teacher_preds is not None:
            for i in range(N):
                errors = []
                img_gt = bboxes_gt[batch_idx_gt == i]
                img_cls_gt = cls_gt[batch_idx_gt == i] if cls_gt is not None else None

                # pred shape (nc + 4, 8400)
                pred = teacher_preds[i]
                boxes = pred[:4, :].T  # (8400, 4)
                scores, class_ids = pred[4:, :].max(0)  # (8400,)

                # Filter by confidence
                conf_mask = scores > self.teacher_pred_conf
                det_boxes = boxes[conf_mask]
                det_cls = class_ids[conf_mask]

                if self.class_mapping is not None:
                    # Filter and remap teacher classes to student classes
                    mask_in_mapping = torch.zeros_like(det_cls, dtype=torch.bool)
                    remapped_cls = det_cls.clone()
                    for k, v in self.class_mapping.items():
                        k_mask = det_cls == int(k)
                        mask_in_mapping |= k_mask
                        remapped_cls[k_mask] = v

                    det_boxes = det_boxes[mask_in_mapping]
                    det_cls = remapped_cls[mask_in_mapping]

                # Normalize teacher boxes to [0, 1] (scaling depends on imgsz used in forward)
                det_boxes_norm = det_boxes.clone()
                det_boxes_norm[:, [0, 2]] /= img_w
                det_boxes_norm[:, [1, 3]] /= img_h

                if len(img_gt) > 0:
                    if len(det_boxes_norm) > 0:
                        # Match with IoU
                        iou = self._bbox_iou_batch(det_boxes_norm, img_gt)  # (K, M)

                        if img_cls_gt is not None:
                            # Only match if classes are consistent (after mapping)
                            # det_cls: (K,), img_cls_gt: (M, 1) or (M,)
                            cls_match = det_cls.unsqueeze(1) == img_cls_gt.view(1, -1)
                            iou = iou * cls_match.float()

                        # FPs: Detections with no GT match
                        max_iou_det, _ = iou.max(1)
                        fp_mask = max_iou_det < 0.5
                        if fp_mask.any():
                            errors.append((det_boxes_norm[fp_mask], 0.0))

                        # FNs: GTs with no detection match
                        max_iou_gt, _ = iou.max(0)
                        fn_mask = max_iou_gt < 0.5
                        if fn_mask.any():
                            errors.append((img_gt[fn_mask], 0.0))
                    else:
                        # Teacher missed all GTs (FNs)
                        errors.append((img_gt, 0.0))
                else:
                    # No GT, everything teacher detected is FP
                    if len(det_boxes_norm) > 0:
                        errors.append((det_boxes_norm, 0.0))

                batch_errors.append(errors)

        masks = []
        for out in self.student_outputs:
            if isinstance(out, list):
                out = out[0]
            _, _, H, W = out.shape
            # Initialize with 1.0: distill everything else
            mask = torch.ones((N, 1, H, W), device=device)

            for i in range(N):
                if teacher_preds is not None:
                    # Apply pre-calculated exclusion zones
                    for err_boxes, val in batch_errors[i]:
                        self._apply_bbox_mask(mask[i], err_boxes, H, W, value=val)
                else:
                    # Fallback to previous logic: only distill GT regions if teacher_preds missing
                    img_gt = bboxes_gt[batch_idx_gt == i]
                    if len(img_gt) > 0:
                        # Here we initialize with 0 and fill GT with 1
                        mask[i] = 0.0
                        self._apply_bbox_mask(mask[i], img_gt, H, W, value=1.0)

            masks.append(mask)
        return masks

    def _bbox_iou_batch(self, boxes1, boxes2):
        """boxes1: (K, 4) xywh normalized or pixels boxes2: (M, 4) xywh normalized or pixels returns: (K, M).
        """
        # Convert to xyxy
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            boxes1[:, 0] - boxes1[:, 2] / 2,
            boxes1[:, 1] - boxes1[:, 3] / 2,
            boxes1[:, 0] + boxes1[:, 2] / 2,
            boxes1[:, 1] + boxes1[:, 3] / 2,
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            boxes2[:, 0] - boxes2[:, 2] / 2,
            boxes2[:, 1] - boxes2[:, 3] / 2,
            boxes2[:, 0] + boxes2[:, 2] / 2,
            boxes2[:, 1] + boxes2[:, 3] / 2,
        )

        # Intersection
        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(0)
        inter_h = (inter_y2 - inter_y1).clamp(0)
        inter_area = inter_w * inter_h

        # Union
        area1 = (boxes1[:, 2] * boxes1[:, 3]).unsqueeze(1)
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union = area1 + area2 - inter_area

        return inter_area / (union + 1e-6)

    def _apply_bbox_mask(self, mask, bboxes, H, W, value=0.0):
        """Applies values to mask regions corresponding to bboxes."""
        for bbox in bboxes:
            xc, yc, bw, bh = bbox
            x1 = int((xc - bw / 2) * W)
            y1 = int((yc - bh / 2) * H)
            x2 = int(torch.ceil((xc + bw / 2) * W))
            y2 = int(torch.ceil((yc + bh / 2) * H))

            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W, x2))
            y2 = max(0, min(H, y2))

            if x2 > x1 and y2 > y1:
                mask[0, y1:y2, x1:x2] = value

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle.clear()
