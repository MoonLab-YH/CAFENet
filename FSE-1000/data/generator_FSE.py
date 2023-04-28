import os
import numpy as np
import random
import torch
from PIL import Image


class FSE_generator(object):

    def __init__(self, path_dir, nb_classes=1, n_shot = 5, n_epoch=1, train=True):
        super(FSE_generator, self).__init__()
        self.base_dir = path_dir
        self.nb_classes = nb_classes
        self.categories = os.listdir(path_dir)
        self.max_iter = n_epoch
        self.num_iter = 0
        self.nb_shot = n_shot
        self.train = train

    def _load_data(self, data_file):
        data_dict = np.load(data_file)
        return {key: np.array(val) for (key, val) in data_dict.items()}

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labelSeg, label1 = self.sample()

            return (self.num_iter - 1), images, labelSeg, label1
        else:
            raise StopIteration()

    def sample(self): # for no_BG data
        sampled_class = random.sample(self.categories, self.nb_classes)  # list of keys
        images_support = []; images_query = [];  label_seg_support = []; label_seg_query = []
        label1_support = [] ; label1_query = []
        for _class in sampled_class:  # k=label->change nb_samples_per_class with different k
            class_dir = os.path.join(self.base_dir, _class)
            supportIdx = random.sample(range(1,11), 5)
            for idx in range(1,11):
                img = Image.open(os.path.join(class_dir, str(idx) + '.jpg'))
                labelSeg = Image.open(os.path.join(class_dir, str(idx)+'.png'))
                label1 = Image.open(os.path.join(class_dir, str(idx)+'_edge.jpg'))

                if self.base_dir.find("train") > 0:
                    degree = 90*random.randrange(0,4)
                    img = img.rotate(degree)
                    labelSeg = labelSeg.rotate(degree)
                    label1 = label1.rotate(degree)

                imgnp = np.array(img).transpose([2,0,1])
                labelSegnp = np.array(labelSeg)
                label1np = np.array(label1)
                if idx in supportIdx:
                    images_support.append(imgnp);    label_seg_support.append(labelSegnp)
                    label1_support.append(label1np);
                else:
                    images_query.append(imgnp);      label_seg_query.append(labelSegnp)
                    label1_query.append(label1np);
        try:
            return torch.Tensor(np.stack(images_support +images_query)), torch.Tensor(np.stack(label_seg_support + label_seg_query))\
                ,torch.Tensor(np.stack(label1_support+label1_query))
        except ValueError:
            print('class name ', _class)