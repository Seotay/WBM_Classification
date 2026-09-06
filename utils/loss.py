import numpy as np
import torch
import torch.nn as nn

def imbalance_weight(distribution, num_classes=9, none_class_idx=8, none_alpha=0.1, device=None):
    n_k = np.array([distribution[i]["count"] for i in range(num_classes)], dtype=np.float32)
    n = n_k.sum()

    freq = n_k / n
    inv_freq = 1.0 / freq    

    
    alpha_k = inv_freq / inv_freq.sum() # [0.1239824  0.1239824  0.1239824  0.1239824  0.1239824  0.1239824 0.1239824  0.1239824  0.00814083]

    alpha_normalized = alpha_k.copy()
    alpha_normalized[none_class_idx] = none_alpha # [0.1239824  0.1239824  0.1239824  0.1239824  0.1239824  0.1239824 0.1239824  0.1239824  0.1]

    
    old_rest_sum = alpha_k[:none_class_idx].sum()
    new_rest_sum = 1.0 - none_alpha


    r_k = new_rest_sum * (1/old_rest_sum)

    for i in range(num_classes):
        if i != none_class_idx:
            alpha_normalized[i] = alpha_k[i] * r_k


    return alpha_normalized


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction


        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = self.CE(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
                focal_loss = focal_loss * alpha_t
            else:
                focal_loss = focal_loss * self.alpha

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        