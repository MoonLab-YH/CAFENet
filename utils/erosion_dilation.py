from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def erosion_dilation(img, kernel_size, size = None, size_query=5):
    imgnp = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=True)
    imgnp = imgnp[:,0,:,:].detach().cpu().numpy() # [5,size,size]
    resultnp = np.zeros((size_query,1,size,size), np.float)

    for i in range(size_query):
        imgtmp = imgnp[i]
        imgtmp[imgtmp<0.4] = 0

        imgten = torch.tensor(imgtmp)
        pad = int((kernel_size-1)/2)
        avged = F.avg_pool2d(imgten.view(1,1,*(imgten.shape)), kernel_size=kernel_size, stride = 1, padding=pad)


        minus = (avged[0,0]-imgten).abs().abs()
        minus = np.expand_dims(minus, axis=0)

        resultnp[i] = minus

    return torch.tensor(resultnp).float().cuda()

