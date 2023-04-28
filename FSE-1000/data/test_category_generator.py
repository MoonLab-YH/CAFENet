import numpy as np
import os
from PIL import Image
import torch

class test_sbd5i_generator():
    def __init__(self, path_dir, n_epoch, n_sample, split_number, category, train=True, rotate=True):
        super(test_sbd5i_generator, self).__init__()
        self.path_dir = path_dir
        self.edge_dir = '/drive1/YH/edges/512_pad_edges'
        self.n_epoch = n_epoch
        self.n_sample=n_sample
        self.split_number = split_number
        self.train=train
        self.num_iter = 0
        self.split_list = [0,1,2,3]
        self.split_list.remove(self.split_number) # 이 split은 test로 사용됨. 각 split에 대해 돌려야해서 총 4번 돌려야함.
        self.class_list = ['aeroplane','bicycle','bird','boat','bottle','bus','car',
                           'cat','chair','cow','diningtable','dog','horse','motorbike',
                           'person','pottedplant','sheep','sofa','train','tvmonitor']
        self.size_support = 5
        self.rotate=rotate
        self.category = category

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num_iter < self.n_epoch:
            self.num_iter += 1
            split = self.split_number

            img_support = [];  img_query = [];  Seg_support = []
            Seg_query = [];  GT_support = [];  GT_query = []

            category = self.category
            img_dir = os.path.join(self.path_dir, str(split), category, 'origin')
            GT_dir = os.path.join(self.edge_dir, str(split), category, 'groundtruth')
            Seg_dir = os.path.join(self.path_dir, str(split), category, 'segmentation')

            filenames = os.listdir(img_dir)
            filelen = len(filenames)

            sample_indices = np.random.permutation(filelen)[:self.n_sample]
            support_indices = np.random.permutation(self.n_sample)

            for i, sample_idx in enumerate(sample_indices):
                img = Image.open(os.path.join(img_dir,filenames[sample_idx]))
                GT = Image.open(os.path.join(GT_dir,filenames[sample_idx]))
                Seg = Image.open(os.path.join(Seg_dir,filenames[sample_idx]))

                imgnp = np.array(img)
                GTnp = np.array(GT)
                if len(GTnp.shape) >= 3:
                    GTnp = GTnp[:,:,0]
                Segnp = np.array(Seg)

                if support_indices[i] < self.size_support:
                    img_support.append(imgnp)
                    GT_support.append(GTnp)
                    Seg_support.append(Segnp)
                else:
                    img_query.append(imgnp)
                    GT_query.append(GTnp)
                    Seg_query.append(Segnp)
            try:
                imgs = torch.tensor(np.stack(img_support+img_query).transpose([0,3,1,2])).float()
                GTs = torch.tensor(np.stack(GT_support+GT_query)).float()
            except ValueError:
                for i in range(5):
                    print(GT_support[i].shape)
                for i in range(5):
                    print(GT_query[i].shape)
            Segs = torch.tensor(np.stack(Seg_support+Seg_query)).float()
            return (self.num_iter - 1), imgs, Segs, GTs
        else:
            raise StopIteration()

