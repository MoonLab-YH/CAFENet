import torch
import numpy as np
from sklearn.metrics import average_precision_score
from utils.functions import *
from PIL import Image

class Evaluator():
    def __init__(self, num_class=2):
        self.num_class = num_class
        self.eps = 1e-6
        self.thresholds = [0.01*i for i in range(10,41)]
        self.confusion_matrix = torch.zeros((len(self.thresholds), self.num_class, self.num_class)).cuda().long() # (2) is scalar, (2,) is tuple

    def Precision(self):
        precision = self.confusion_matrix[:,1,1] / (self.confusion_matrix[:,0,1] + self.confusion_matrix[:,1,1] + self.eps)
        return precision

    def Recall(self):
        recall = self.confusion_matrix[:,1,1] / (self.confusion_matrix[:,1,0] + self.confusion_matrix[:,1,1] + self.eps)
        return recall

    def F1_measure(self):
        F1 = 2*self.Precision()*self.Recall() / (self.Precision()+self.Recall()+self.eps)
        return F1

    def AP(self, prediction, target):
        ap = average_precision_score(target, prediction)
        return ap

    def generate_matrix(self, prediction, target):

        label = (self.num_class * target.long() + prediction.long()).view(-1)
        # gt_image[mask] returns flattened vector
        # make confunsion matrix's element. unique value for each prediction label and each target label
        count = torch.bincount(label, minlength=self.num_class ** 2)  # same as count function, but consider unhappened event
        confu_matrix = count.reshape(self.num_class, self.num_class)
        return confu_matrix

    def add_batch(self, prediction, target):
        pred, tar = prediction.clone(), target.clone()
        assert pred.shape == tar.shape

        for idx, threshold in enumerate(self.thresholds):
            pred_tmp = pred.clone()
            pred_tmp[pred_tmp > threshold] = 1
            pred_tmp[pred_tmp < threshold] = 0

            self.confusion_matrix[idx]  = self.confusion_matrix[idx] + self.generate_matrix(pred_tmp, tar)
        a=2
        # performance of model is computed as the form of fraction number --> keep summing confunsion matrix is safe

    # def reset(self):
    #     self.confusion_matrix = torch.zeros((2,)*2).cuda().long()