from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import os
import time
import random
from sklearn.metrics import average_precision_score

def generate_label(labels):
    with torch.no_grad():
        labels_tmp = labels.view(labels.size(0), 1, labels.size(1), labels.size(2)).float()
        labels_tmp = F.interpolate(labels_tmp, size=[int(labels_tmp.size(2) / 16), int(labels_tmp.size(3) / 16)],
                                   mode='bilinear')
        labels_tmp = torch.round(labels_tmp)

        labels_tmp2 = 1 - labels_tmp
        labels_gen = torch.cat((labels_tmp2, labels_tmp), dim=1)

    return labels_gen

def split_query(prototype, query_set, temp):

    W=H=query_set.shape[2]
    FG_proto = prototype[1].unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)  # [1,512,1,1]
    BG_proto = prototype[0].unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)  # [1,512,1,1]
    FG_corr = F.cosine_similarity(query_set, FG_proto).unsqueeze(dim=1)  # [10,1,20,20]
    BG_corr = F.cosine_similarity(query_set, BG_proto).unsqueeze(dim=1)  # [10,1,20,20]

    FG_corr = F.softmax(FG_corr[:, 0, :, :].view(10, -1) * temp, dim=1).view(10, 1, W, H)
    BG_corr = F.softmax(BG_corr[:, 0, :, :].view(10, -1) * temp, dim=1).view(10, 1, W, H)

    query_FG = query_set * FG_corr;
    query_BG = query_set * BG_corr;

    return query_FG, query_BG

def Dist_Prediction_MultiHead_PN(H1,H2,H3,H4,S1,S2,S3,S4,query_set,prototype, temp):
    W=H1.shape[2] # [10,128,20,20]
    batch_size = H1.shape[0] # [2,512]
    n_channel = H1.shape[1]
    H = query_set # [10,512,20,20]
    P1 = prototype.index_select(1, S1)
    P2 = prototype.index_select(1, S2)
    P3 = prototype.index_select(1, S3)
    P4 = prototype.index_select(1, S4)


    H1_flattened = H1.permute([0,2,3,1]).contiguous().view(-1,1,n_channel) # [10*20*20,1,128]
    H2_flattened = H2.permute([0,2,3,1]).contiguous().view(-1,1,n_channel) # [10*20*20,1,128]
    H3_flattened = H3.permute([0,2,3,1]).contiguous().view(-1,1,n_channel) # [10*20*20,1,128]
    H4_flattened = H4.permute([0,2,3,1]).contiguous().view(-1,1,n_channel) # [10*20*20,1,128]
    H_flattened = H.permute([0,2,3,1]).contiguous().view(-1,1,n_channel*4) # [10*20*20,1,512]

    H1_dist = (H1_flattened - P1).pow(2).sum(dim=2) # [10*20*20,2], 0 BG, 1 FG
    H2_dist = (H2_flattened - P2).pow(2).sum(dim=2) # [10*20*20,2], 0 BG, 1 FG
    H3_dist = (H3_flattened - P3).pow(2).sum(dim=2) # [10*20*20,2], 0 BG, 1 FG
    H4_dist = (H4_flattened - P4).pow(2).sum(dim=2) # [10*20*20,2], 0 BG, 1 FG
    H_dist = (H_flattened - prototype).pow(2).sum(dim=2) # [10*20*20,2], 0 BG, 1 FG

    H1_dist = H1_dist.view(batch_size,W,W,2).contiguous().permute([0,3,1,2]) # [10,2,20,20]
    H2_dist = H2_dist.view(batch_size,W,W,2).contiguous().permute([0,3,1,2]) # [10,2,20,20]
    H3_dist = H3_dist.view(batch_size,W,W,2).contiguous().permute([0,3,1,2]) # [10,2,20,20]
    H4_dist = H4_dist.view(batch_size,W,W,2).contiguous().permute([0,3,1,2]) # [10,2,20,20]
    H_dist = H_dist.view(batch_size,W,W,2).contiguous().permute([0,3,1,2]) # [10,2,20,20]

    pred1 = F.softmax(-H1_dist*temp, dim=1)
    pred2 = F.softmax(-H2_dist*temp, dim=1)
    pred3 = F.softmax(-H3_dist*temp, dim=1)
    pred4 = F.softmax(-H4_dist*temp, dim=1)
    pred = F.softmax(-H_dist*temp, dim=1)

    return pred1, pred2, pred3, pred4, pred

def Dist_Prediction_PN(query_set, prototype, temp):
    W=H=query_set.shape[2] # [10,512,20,20]
    batch_size = query_set.shape[0] # [2,512]
    n_channel = query_set.shape[1]
    query_flattened = query_set.permute([0,2,3,1]).contiguous().view(-1,1,n_channel)
    dist = (query_flattened - prototype).pow(2).sum(dim=2) # [10*20*20,2], 0 BG, 1 FG
    dist_to_BG = dist[:,0].view(batch_size,1,W,H)
    dist_to_FG = dist[:,1].view(batch_size,1,W,H)
    pred = torch.cat((-dist_to_BG, -dist_to_FG), dim=1)  # [10,2,20,20]
    pred = F.softmax(pred * temp, dim=1) # [10,2,20,20]

    return  pred

def visualize(imgten, path, color=True, threshold = False, size=None, reverse = False):
    if color: # input should be [C,W,H]
        if imgten.size(0) == 3:
            imgten = F.interpolate(imgten.unsqueeze(dim=0), size=(320, 320), mode='bilinear', align_corners=True)
            imgten = imgten[0].permute([1,2,0])
            imgnp = imgten.detach().cpu().numpy()
        else:
            imgten = imgten.unsqueeze(dim=0).unsqueeze(dim=0).float()
            imgten = F.interpolate(imgten, size=(320, 320), mode='bilinear', align_corners=True)
            imgnp = imgten[0,0].detach().cpu().numpy()
        imgnp = np.interp(imgnp, (imgnp.min(), imgnp.max()), (0,255)).astype(np.uint8)
        # imgnp = 255 - imgnp
        img = Image.fromarray(imgnp)
        img.save(path)
    else: #grayscale, input should be [W,H]
        imgten = imgten.unsqueeze(dim=0).unsqueeze(dim=0).float()
        imgten = F.interpolate(imgten, size=(320,320), mode='bilinear', align_corners=True)
        imgnp = imgten[0,0].detach().cpu().numpy()
        imgnp = np.interp(imgnp, (imgnp.min(), imgnp.max()), (0,255)).astype(np.uint8)
        if threshold:
            imgnp[imgnp<threshold] = 0; imgnp[imgnp>=threshold] = 255
        if reverse:
            imgnp = 255 - imgnp
        img = Image.fromarray(imgnp)
        img.save(path)

def DeleteContent(path):
    eval_list = os.listdir(path)
    for i in eval_list:
        os.remove(os.path.join(path,i))

def Binarize(input, threshold):
    output = input.clone()
    output[output < threshold] = 0
    output[output >= threshold] = 1
    output = output.long()
    return output

def compute_AP(pred, target): # imagewise
    query_size = pred.shape[0]
    AP_sum = 0
    div = query_size
    for img_idx in range(query_size):
        if len(torch.unique(target[img_idx])) == 1:
            div -= 1
            continue
        eval_targets = target[img_idx].view(-1).cpu().numpy()  # [224,224]
        eval_preds = pred[img_idx].contiguous().view(-1).cpu().numpy()  # [224,224]
        AP_sum += average_precision_score(eval_targets, eval_preds)

    AP = AP_sum / div
    return AP

def Spliiter(qry):
    S = [set(range(128*i,128*(i+1))) for i in range(4)]
    out = [set([]) for _ in range(4)]
    for idxMy, _ in enumerate(S):
        for idxOut in range(4):
            if idxOut == idxMy:
                continue
            mini = random.sample(S[idxMy], 12)
            S[idxMy] = S[idxMy] - set(mini)
            out[idxOut] |= set(mini)
    for idxMy, _ in enumerate(S):
        S[idxMy] |= out[idxMy]
        S[idxMy] = torch.tensor(list(S[idxMy])).cuda()
    [H1, H2, H3, H4] = [qry.index_select(1, S[i]) for i in range(len(S))]

    return H1,H2,H3,H4,S[0],S[1],S[2],S[3]


