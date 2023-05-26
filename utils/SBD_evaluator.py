import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from PIL import Image
from utils.functions import *

class Evaluator():
    def __init__(self, num_class=2, tolerance = 0.01):
        self.num_class = num_class
        self.eps = 1e-6
        self.thresholds = [0.01*i for i in range(1,100)]
        self.device = torch.device('cuda:0')
        self.confusion_matrix = torch.zeros((len(self.thresholds), self.num_class, self.num_class)).long().to(self.device)
        self.single_confumat = torch.zeros((len(self.thresholds), self.num_class, self.num_class)).long().to(self.device)
        self.tolerance = tolerance

    def Precision(self):
        precision = self.confusion_matrix[:,1,1] / (self.confusion_matrix[:,0,1] + self.confusion_matrix[:,1,1] + self.eps)
        return precision

    def Recall(self):
        recall = self.confusion_matrix[:,1,1] / (self.confusion_matrix[:,1,0] + self.confusion_matrix[:,1,1] + self.eps)
        return recall

    def Precision_single(self):
        precision = self.single_confumat[:,1,1] / (self.single_confumat[:,0,1] + self.single_confumat[:,1,1] + self.eps)
        return precision

    def Recall_single(self):
        recall = self.single_confumat[:,1,1] / (self.single_confumat[:,1,0] + self.single_confumat[:,1,1] + self.eps)
        return recall

    def F1_measure(self):
        F1 = 2*self.Precision()*self.Recall() / (self.Precision()+self.Recall()+self.eps)
        return F1

    def AP(self):
        # R_shift = torch.cat((self.Recall().flip(dims=[0])[1:], torch.tensor([1.]).cuda()))
        R_shift = torch.cat((torch.tensor([0.]).cuda(), self.Recall().flip(dims=[0])[:-1]))
        R_width = self.Recall().flip(dims=[0]) - R_shift
        P_height = self.Precision().flip(dims=[0])
        PR = R_width * P_height
        AP = PR.sum()
        return AP

    def generate_matrix(self, prediction, target):
        label = (self.num_class * target.long() + prediction.long()).view(-1)
        count = torch.bincount(label, minlength=self.num_class ** 2)  # same as count function, but consider unhappened event
        confu_matrix = count.reshape(self.num_class, self.num_class).to(self.device)
        return confu_matrix

    def generate_matrix_tolerance(self, prediction, target, diltar, idx):
        TP = ((prediction == 1) & (diltar == 1)).sum().item()
        TN = ((prediction == 0) & (target == 0)).sum().item()
        FP = ((prediction == 1) & (diltar == 0)).sum().item()
        FN = ((prediction == 0) & (target == 1)).sum().item()
        confu_matrix = torch.tensor([[TN,FP],[FN,TP]]).cuda()
        self.single_confumat[idx] = confu_matrix
        return confu_matrix

    def MakeKernel(self, tolerance=0.02, H=512, W=512):
        diag = math.sqrt(H**2 + W**2)
        radius = int(diag * tolerance)
        size = 2*radius + 1
        kernel = torch.zeros(size, size).cuda()
        for i in range(size):
            for j in range(size):
                if (radius - i) ** 2 + (radius - j) ** 2 <= radius ** 2:
                    kernel[i][j] = 1
        return kernel, radius

    def add_batch(self, prediction, target, size):
        pred, target = prediction.clone(), target.clone()
        kernel, radius = self.MakeKernel(self.tolerance, *size)
        diltar = F.conv2d(target[None, None].float(), kernel[None,None], padding=radius)[0, 0]
        diltar = torch.clamp(diltar, 0, 1)
        assert pred.shape == target.shape

        for idx, threshold in enumerate(self.thresholds):
            pred_tmp = pred.clone()
            pred_tmp[pred_tmp > threshold] = 1
            pred_tmp[pred_tmp < threshold] = 0
            # visualize(prediction, 'visualization/th_pred.jpg')
            self.confusion_matrix[idx] = self.confusion_matrix[idx] + \
                                         self.generate_matrix_tolerance(pred_tmp, target, diltar, idx)

        AP = self.AP().item()
        return AP
