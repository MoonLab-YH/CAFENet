import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from modeling.CAFENET_decoder import Decoder
from modeling.CAFENET_backbone import resnet34
from modeling.CAFENET_attention import AttMaker
from modeling.HyperParam import HyperParam
from modeling.CAFENET_aspp import ASPP
from modeling.PostEnc_Fusion import Mini_Postconv, backbone1
from utils.functions import *
from utils.erosion_dilation import erosion_dilation

class CAFENet(nn.Module):
    def __init__(self, nb_class, input_size,
                 n_shot, dropout_aspp, dropout_dec, device):

        super(CAFENet, self).__init__()
        self.nb_class = nb_class
        self.input_size = input_size
        self.n_shot = n_shot
        self.encoder = resnet34(pretrained=True)
        self.decoder = Decoder()
        self.attmaker = AttMaker()
        self.PostEncoder = Mini_Postconv()
        self.HyperParam = HyperParam()
        self.backbone1 = backbone1()
        self.ASPP = ASPP()

        self.device = device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.HyperParam.to(self.device)
        self.PostEncoder.to(self.device)
        self.backbone1.to(self.device)
        self.ASPP.to(self.device)

    def set_optimizer(self, learning_rate, weight_decay_rate1, weight_decay_rate2, lrstep, decay_rate):
        self.optimizer1 = optim.AdamW(
            list(self.HyperParam.parameters()) + list(self.backbone1.parameters()) +
            list(self.PostEncoder.parameters()) + list(self.attmaker.parameters()) + list(self.ASPP.parameters()),
            lr=learning_rate)
        self.scheduler1 = optim.lr_scheduler.StepLR(self.optimizer1, step_size=lrstep, gamma=decay_rate)

        self.optimizer2 = optim.AdamW(list(self.encoder.parameters()), lr=learning_rate / 10)
        self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer2, step_size=lrstep, gamma=decay_rate)

        self.optimizer3 = optim.AdamW(list(self.decoder.parameters()), lr=learning_rate * 20)
        self.scheduler3 = optim.lr_scheduler.StepLR(self.optimizer3, step_size=lrstep, gamma=decay_rate)

    def compute_prototypes(self, labels_gen, data_aspp):
        for i in range(self.nb_class + 1):
            label_tmp = labels_gen[:, i] # [5,2,20,20] -> [5,20,20]
            label_tmp = label_tmp.view(label_tmp.size(0), 1, label_tmp.size(1), label_tmp.size(2)) # [5,1,20,20]
            data_label = data_aspp * label_tmp
            if i == 0:
                prototype = data_label.mean((2, 3)) # spatial mean
                prototype = prototype.mean(0, keepdim=True) # batch mean
            else:
                prototype_tmp = data_label.mean((2, 3))
                prototype_tmp = prototype_tmp.mean(0, keepdim=True)
                prototype = torch.cat((prototype, prototype_tmp))
        return prototype


    def Train(self, images, labels, GT):
        self.train()
        labels = Binarize(labels, threshold=20)
        GT = Binarize(GT, threshold=50)[self.n_shot:]

        feat, E1, E2, E3, E4 = self.encoder(images)  # 512,64,64,128
        support_set = feat[:self.n_shot]
        query_set = feat[self.n_shot:]  # [10,512,20,20]
        H1,H2,H3,H4,S1,S2,S3,S4 = Spliiter(query_set)
        # H1, H2, H3, H4 = query_set[:, :128], query_set[:, 128:256], query_set[:, 256:384], query_set[:, 384:]

        labels_gen = generate_label(labels)
        labels_support = labels_gen[:self.nb_class * self.n_shot]
        labels_query = labels_gen[self.nb_class * self.n_shot:]  # [10,2,20,20]
        prototype = self.compute_prototypes(labels_support, support_set)  # [2,512]

        norm_query = F.normalize(query_set, dim=1)  # [10,512,20,20]
        norm_proto = F.normalize(prototype, dim=1)  # [2,512]

        pred1, pred2, pred3, pred4, pred = Dist_Prediction_MultiHead_PN(H1, H2, H3, H4, S1, S2, S3, S4, query_set,
                                                                        norm_proto, self.HyperParam.tau_dist)

        loss_MSMR = F.mse_loss(pred1, labels_query) + F.mse_loss(pred2, labels_query) + F.mse_loss(pred3, labels_query) + \
                   F.mse_loss(pred4, labels_query) + F.mse_loss(pred, labels_query)

        E0 = self.backbone1(images)[self.n_shot:]
        E3 = self.ASPP(E3)
        Post_E1, Post_E2, Post_E3, Post_E4 = self.PostEncoder(E1, E2, E3, E4)
        Dec_input_4 = torch.cat((norm_query, Post_E4, pred), dim=1)

        Att4, Att3, Att2, Att1, Att0 = self.attmaker(Dec_input_4, Post_E3, Post_E2, Post_E1, E0)

        Dec_input_4 = Dec_input_4 * (1 + Att4)
        Dec_input_3 = Post_E3 * (1 + Att3)
        Dec_input_2 = Post_E2 * (1 + Att2)
        Dec_input_1 = Post_E1 * (1 + Att1)
        Dec_input_0 = E0 * (1 + Att0)

        dec_image = self.decoder(Dec_input_4, Dec_input_3, Dec_input_2, Dec_input_1, Dec_input_0) # [10,2,80,80]
        loss_edge = F.cross_entropy(dec_image, GT, weight=torch.tensor([1., 1.]).cuda())

        dec_pred = F.softmax(dec_image, dim=1)[:, 1, :, :]
        dice_loss = (dec_pred.pow(2).sum() + GT.pow(2).sum()) / (2 *(dec_pred * GT).sum())

        loss = 0.1*loss_MSMR + loss_edge + dice_loss
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        loss.backward()
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer3.step()
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler3.step()
        return loss_MSMR.data, loss_edge, dice_loss.data


    def evaluate(self, images, labels, GT):
        self.eval()
        labels = Binarize(labels, threshold=127)
        GT = Binarize(GT, threshold=127)[self.n_shot:]

        with torch.no_grad():
            feat, E1, E2, E3, E4 = self.encoder(images)  # 512,64,64,128
            support_set = feat[:self.n_shot]
            query_set = feat[self.n_shot:]  # [10,512,20,20]
            H1,H2,H3,H4,S1,S2,S3,S4 = Spliiter(query_set)

            labels_gen = generate_label(labels)
            labels_support = labels_gen[:self.nb_class * self.n_shot]
            labels_query = labels_gen[self.nb_class * self.n_shot:]  # [10,2,20,20]
            prototype = self.compute_prototypes(labels_support, support_set)  # [2,512]

            norm_query = F.normalize(query_set, dim=1)  # [10,512,20,20]
            norm_proto = F.normalize(prototype, dim=1)  # [2,512]

            pred1, pred2, pred3, pred4, pred = Dist_Prediction_MultiHead_PN(H1, H2, H3, H4, S1, S2, S3, S4, query_set,
                                                                            norm_proto, self.HyperParam.tau_dist)
            loss_MSMR = F.mse_loss(pred1, labels_query) + F.mse_loss(pred2, labels_query) + F.mse_loss(pred3,labels_query) +\
                       F.mse_loss(pred4, labels_query) + F.mse_loss(pred, labels_query)

            E0 = self.backbone1(images)[self.n_shot:]
            E3 = self.ASPP(E3)
            Post_E1, Post_E2, Post_E3, Post_E4 = self.PostEncoder(E1, E2, E3, E4)
            Dec_input_4 = torch.cat((norm_query, Post_E4, pred), dim=1)

            Att4, Att3, Att2, Att1, Att0 = self.attmaker(Dec_input_4, Post_E3, Post_E2, Post_E1, E0)

            Dec_input_4 = Dec_input_4 * (1 + Att4)
            Dec_input_3 = Post_E3 * (1 + Att3)
            Dec_input_2 = Post_E2 * (1 + Att2)
            Dec_input_1 = Post_E1 * (1 + Att1)
            Dec_input_0 = E0 * (1 + Att0)

            dec_image = self.decoder(Dec_input_4, Dec_input_3, Dec_input_2, Dec_input_1, Dec_input_0)  # [10,2,80,80]
            loss_edge = F.cross_entropy(dec_image, GT, weight=torch.tensor([1., 1.]).cuda())

            dec_pred = F.softmax(dec_image, dim=1)[:, 1, :, :]
            dice_loss = (dec_pred.pow(2).sum() + GT.pow(2).sum()) / (2 * (dec_pred * GT).sum())

            AP = compute_AP(dec_pred, GT)

            return loss_MSMR.data, loss_edge.data, dice_loss.data, dec_pred, pred, AP

