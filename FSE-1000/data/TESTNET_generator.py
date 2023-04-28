"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot
"""
import os
import numpy as np
import random
import torch
from PIL import Image


class FSS_generator(object):
    """miniImageNetGenerator

    Args:
        data_file (str): 'data/Imagenet/train.npz' or 'data/Imagenet/test.npz'
        nb_classes (int): number of classes in an episode
        nb_samples_per_class (int): nuber of samples per class in an episode
        max_iter (int): max number of episode generation
    """

    def __init__(self, base_dir, nb_classes=1, nb_shot = 5, max_iter=None, istrain=True):
        super(FSS_generator, self).__init__()
        self.base_dir = base_dir
        self.nb_classes = nb_classes
        # with open(os.path.join(base_dir, 'all.txt')) as f:
        #     self.categories = f.read().splitlines()
        self.categories = os.listdir(base_dir)
        self.max_iter = max_iter
        self.num_iter = 0
        self.nb_shot = nb_shot
        self.istrain = istrain

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

            return (self.num_iter - 1), (images, labelSeg, label1)
        else:
            raise StopIteration()

    def sample(self): # for no_BG data
        sampled_class = random.sample(self.categories, self.nb_classes)  # list of keys
        images_support = []; images_query = [];  label_seg_support = []; label_seg_query = []
        label1_support = [] ; label1_query = [];
        for _class in sampled_class:  # k=label->change nb_samples_per_class with different k
            class_dir = os.path.join(self.base_dir, _class)
            supportIdx = random.sample(range(1,11), 5)
            for idx in range(1,11):
                img = Image.open(os.path.join(class_dir, str(idx) + '.jpg'))
                imgnp = np.array(img).transpose([2,0,1])
                labelSeg = Image.open(os.path.join(class_dir, str(idx)+'.png'))
                labelSegnp = np.array(labelSeg)
                label1 = Image.open(os.path.join(class_dir, str(idx)+'_edge.jpg'))
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


def Make_All_txt(base_dir):
    if not os.path.exists(os.path.join(base_dir,'all.txt')):
        categories = os.listdir(base_dir)
        with open(os.path.join(base_dir,'all.txt'), "w") as f:
            for category in categories:
                f.write(category+'\n')

def Equalize_Size(base_dir):
    class_num = 0
    errCnt = 0
    categories = os.listdir(base_dir)
    shape2D = (224,224); shape3D = (224,224,3)
    for category in categories:
        if not os.path.isdir(os.path.join(base_dir,category)):
            continue
        class_num += 1
        if class_num % 50 == 0:
            print('processing %d / 1000' % class_num)
        class_dir = os.path.join(base_dir,category)
        filenames = os.listdir(class_dir)
        for filename in filenames:
            if os.path.splitext(filename)[-1] == '.png' or os.path.splitext(filename)[-1] == '.jpg':
                img = Image.open(os.path.join(class_dir,filename))
                imgnp = np.array(img)
                if imgnp.shape[:2] != shape2D and imgnp.shape[:2] != shape3D:
                    errCnt += 1
                    img = img.resize((224,224))
                    img.save(os.path.join(class_dir, filename))
                    print(category)
    print('total %d images is not size of (3,224,224) and changed to (3,224,224)' % errCnt)

if __name__=='__main__':
    base_dir = '/drive2/data/SBD-5i/'
    Equalize_Size(base_dir)

