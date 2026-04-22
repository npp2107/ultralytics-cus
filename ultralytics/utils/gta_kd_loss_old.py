from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    Modified for Ground Truth Aware (GTA) distillation.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, y_s, y_t, masks=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
            masks (list, optional): List of Ground Truth masks with shape (N, 1, H, W).
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            s_flat = s.view(N, C, -1)
            t_flat = t.view(N, C, -1)

            if masks is not None and idx < len(masks):
                mask = masks[idx].view(N, 1, -1) # (N, 1, HW)
                
                # Mask out background by setting to -1e9 before softmax to ignore those regions
                s_flat_m = s_flat.clone()
                t_flat_m = t_flat.clone()
                
                # Check for empty masks
                has_pixel = (mask.sum(dim=2) > 0).float() # (N, 1)
                
                # Apply mask: set non-foreground pixels to a very low value
                s_flat_m[mask.expand_as(s_flat_m) == 0] = -1e9
                t_flat_m[mask.expand_as(t_flat_m) == 0] = -1e9
                
                softmax_pred_T = F.softmax(t_flat_m / self.tau, dim=2)
                logsoftmax = torch.nn.LogSoftmax(dim=2)
                
                cost = (softmax_pred_T * logsoftmax(t_flat_m / self.tau) -
                        softmax_pred_T * logsoftmax(s_flat_m / self.tau)) * (self.tau ** 2)
                
                # Average over channels and batch for those images that have GT pixels
                channel_cost = (cost.sum(dim=2) * has_pixel).sum(dim=1) / (C + 1e-6) # (N,)
                batch_cost = channel_cost.sum() / (has_pixel.sum() + 1e-6)
                losses.append(batch_cost)
            else:
                # Standard CWD
                softmax_pred_T = F.softmax(t_flat / self.tau, dim=2)
                logsoftmax = torch.nn.LogSoftmax(dim=2)
                cost = torch.sum(
                    softmax_pred_T * logsoftmax(t_flat / self.tau) -
                    softmax_pred_T * logsoftmax(s_flat / self.tau)) * (self.tau ** 2)
                losses.append(cost / (C * N))
                
        loss = sum(losses)
        return loss

class MGDLoss(nn.Module):
    """Masked Generative Distillation (MGD) loss with Ground Truth Awareness."""
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)
            ).to(device) for channel in teacher_channels
        ])

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
        N, C, H, W = preds_T.shape
        device = preds_S.device
        
        # MGD random masking for generation task
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        if mask is not None:
            # Distill only on Ground Truth Regions (GTA)
            # mask is (N, 1, H, W)
            loss = (new_fea - preds_T)**2 * mask
            dis_loss = loss.sum() / N
        else:
            loss_mse = nn.MSELoss(reduction='sum')
            dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.align_module = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        
        for s_chan, t_chan in zip(channels_s, channels_t):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(t_chan, affine=False)
            ).to(device)
            self.align_module.append(align)
            
        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))

        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t, masks=None):
        if len(y_s) != len(y_t):
            y_t = y_t[len(y_t) // 2:]

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

class DistillationLoss:
    """Ground Truth Aware Distillation Loss.
    Filters distillation to focus on regions corresponding to Ground Truth boxes.
    """
    def __init__(self, models, modelt, distiller="CWDLoss"):
        self.distiller = distiller
        self.s_layers = ["6", "8", "13", "16", "19", "22"]
        self.t_layers = ["6", "8", "13", "16", "19", "22"]
        self.models = models 
        self.modelt = modelt

        device = next(models.parameters()).device
        # Init warm up
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640).to(device)
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
                    if hasattr(ml, 'conv'):
                        self.channels_t.append(ml.conv.out_channels)
                        self.teacher_module_pairs.append(ml)

        for name, ml in self.models.named_modules():
            if name and name.startswith("model."):
                parts = name.split(".")
                if len(parts) >= 3 and parts[1] in self.s_layers and "cv2" in parts[2]:
                    if hasattr(ml, 'conv'):
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

    def get_loss(self, batch=None):
        """Calculate distillation loss.
        Args:
            batch (dict, optional): Training batch containing 'bboxes' and 'batch_idx' for GT-aware masking.
        """
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)
        
        if len(self.teacher_outputs) != len(self.student_outputs):
            print(f"Warning: Mismatched outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.student_outputs)}")
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)
        
        masks = None
        if batch is not None:
            masks = self._generate_masks(batch)

        quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=self.teacher_outputs, masks=masks)
        
        if self.distiller.lower() != 'cwdloss':
            quant_loss *= 0.3

        self.teacher_outputs.clear()
        self.student_outputs.clear()
        
        return quant_loss

    def _generate_masks(self, batch):
        """Creates binary masks for regions corresponding to Ground Truth boxes."""
        bboxes = batch.get('bboxes')
        batch_idx = batch.get('batch_idx')
        if bboxes is None or batch_idx is None:
            return None
        
        masks = []
        for out in self.student_outputs:
            # Handle list of tensors vs tensor
            if isinstance(out, list):
                out = out[0]
            N, _, H, W = out.shape
            mask = torch.zeros((N, 1, H, W), device=bboxes.device)
            
            for i in range(N):
                img_bboxes = bboxes[batch_idx == i]
                if len(img_bboxes) == 0:
                    continue
                
                # YOLO format: xywhn (x_center, y_center, width, height normalized)
                xc, yc, bw, bh = img_bboxes.unbind(1)
                x1 = (xc - bw / 2) * W
                y1 = (yc - bh / 2) * H
                x2 = (xc + bw / 2) * W
                y2 = (yc + bh / 2) * H
                
                for b_x1, b_y1, b_x2, b_y2 in zip(x1, y1, x2, y2):
                    ix1, iy1 = int(b_x1), int(b_y1)
                    ix2, iy2 = int(b_x2.ceil()), int(b_y2.ceil())
                    # Clip to grid
                    ix1 = max(0, min(W - 1, ix1))
                    iy1 = max(0, min(H - 1, iy1))
                    ix2 = max(0, min(W, ix2))
                    iy2 = max(0, min(H, iy2))
                    if ix2 > ix1 and iy2 > iy1:
                        mask[i, 0, iy1:iy2, ix1:ix2] = 1.0
            masks.append(mask)
        return masks

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle.clear()
