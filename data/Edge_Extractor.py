import os
import numpy as np

from PIL import Image

base_dir = '../../dataset/fewshot_data/fewshot_data'
#base_dir = 'data/fewshot_data/fewshot_data/fewshot_data'
classes_dir = os.listdir(base_dir)
radius = 3


def Edge_Extractor():
    for n,dir in enumerate(classes_dir):
        class_dir = os.path.join(base_dir, dir)
        filenames = os.listdir(class_dir)
        print('%d / 1000 .. processing %s...' % (n, dir))
        for filename in filenames:
            if os.path.splitext(filename)[-1] == '.png':
                img = Image.open(os.path.join(class_dir,filename))
                imgnp = np.array(img)
                if len(imgnp.shape) == 3:
                    imgnp = imgnp[:,:,0]
                W,H = imgnp.shape[0], imgnp.shape[1]
                img = Image.fromarray(imgnp)
                Edgenp = np.ones((W,H), dtype=np.uint8)
                for i in range(W):
                    for j in range(H):
                        Edgenp[i][j] = 0
                        if imgnp[i][j] >= 127: # (255/2 ~= 127)
                            if i<radius or i>W-radius or j < radius or j>H-radius:
                                continue
                            for r in range(radius):
                                if imgnp[i - r][j - r] < 127 or imgnp[i - r][j + r] < 127 or \
                                        imgnp[i + r][j - r] < 127 or imgnp[i + r][j + r] < 127:
                                    Edgenp[i][j] = 255
                                    break


                Edge = Image.fromarray(Edgenp)
                Edge.save(os.path.join(class_dir,os.path.splitext(filename)[0]+'_edge.jpg'))

def Edge_Deleter():
    for n,dir in enumerate(classes_dir):
        class_dir = os.path.join(base_dir, dir)
        filenames = os.listdir(class_dir)
        print('%d / 1000 .. processing %s...' % (n, dir))
        for filename in filenames:
            if '_edge' in filename:
                os.remove(os.path.join(class_dir, filename))

if __name__ == '__main__':
    Edge_Extractor()