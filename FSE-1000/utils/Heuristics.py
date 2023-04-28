import numpy as np
import torch
from sklearn.metrics import average_precision_score
from utils.functions import *

def Roataion_Inference(images, Segs, GT_input, model):
    H = W = images.shape[3]
    query_size = images.shape[0] - model.module.n_shot
    pred_ROTs = np.ones((4, query_size, H, W))
    GTs = GT_input.clone()
    GTs_tmp = GT_input.clone()
    GTs_tmp[GTs_tmp<50] = 0
    GTs_tmp[GTs_tmp>=50] = 1

    for degree in range(0, 4):
        img = torch.tensor(np.rot90(images.cpu().numpy(), degree, axes=(2, 3)).copy()).float()
        GT = torch.tensor(np.rot90(GTs.cpu().numpy(), degree, axes=(1, 2)).copy()).float()
        Seg = torch.tensor(np.rot90(Segs.cpu().numpy(), degree, axes=(1, 2)).copy()).float()

        img, GT, Seg = img.cuda(), GT.cuda(), Seg.cuda()

        loss_TAP, loss_edge, dice_loss, dec_image, pred_TAP, AP \
            = model.module.evaluate(images=img, labels=Seg, GT=GT)

        if degree == 0:
            original_AP = AP

        pred_ROT = dec_image.cpu().numpy()  # [5,224,224]
        pred_ROT = np.rot90(pred_ROT, 4 - degree, axes=(1, 2))
        pred_ROTs[degree] = pred_ROT

    final_output = np.average(pred_ROTs, axis=0) # [4,5,224,224] -> [5,224,224]

    query_size = final_output.shape[0]
    div = query_size
    AP_sum = 0
    for img_idx in range(query_size):
        if len(torch.unique(GTs_tmp[5 + img_idx])) == 1:
            div -= 1
            continue
        eval_targets = GTs_tmp[5+img_idx].cpu().numpy().reshape(-1)  # [224,224]
        eval_preds = final_output[img_idx].reshape(-1)  # [224,224]
        AP_sum += average_precision_score(eval_targets, eval_preds)
    AP = AP_sum / div

    return torch.tensor(final_output), AP, original_AP, pred_TAP

