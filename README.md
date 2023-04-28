# CAFENet
Official Pytorch implementation for the paper titled "CAFENet: Class-Agnostic Few-Shot Edge Detection Network" presented on BMVC 2021.


Network architecture overview of proposed CAFENet             |  Qualitative Results on the SBD-5i Dataset.
:-------------------------:|:-------------------------:
![Network_Overview](https://user-images.githubusercontent.com/54431060/235067420-baa7b275-21ff-40b0-9a98-ff04a716f1f4.png)  | ![image](https://user-images.githubusercontent.com/54431060/235068437-a6b53f39-ed83-4e99-b00c-ad20c837e6ef.png)

# Abstract
We tackle a novel few-shot learning challenge, few-shot semantic edge detection, aiming to localize boundaries of novel categories using only a few labeled samples.
Reliable boundary information has been shown to boost the performance of semantic segmentation and localization, while also playing a key role in its own right in object reconstruction, image generation and medical imaging. However, existing semantic edge detection techniques require a large amount of labeled data to train a model. To overcome this limitation, we present Class-Agnostic Few-shot Edge detection Network (CAFENet) based on a meta-learning strategy. CAFENet employs a semantic segmentation module in small-scale to compensate for the lack of semantic information in edge labels. To effectively fuse the semantic information and low-level cues, CAFENet also utilizes an attention module which dynamically generates multi-scale attention map, as well as a novel regularization method that splits high-dimensional features into several low-dimensional features and conducts multiple metric learning. Since there are no existing datasets for few-shot semantic edge detection, we construct two new datasets, FSE-1000 and SBD-5i, and evaluate the performance of the proposed CAFENet on them. Extensive simulation results confirm that CAFENet achieves better performance compared to the baseline methods using fine-tuning or few-shot segmentation.


# Environment Info
```
sys.platform: linux

Python: 3.7.7
Pytorch : 1.5.0  
TorchVision: 0.6.1
Cudatoolkit : 10.1.243  
scikit-learn : 0.23.2
numpy : 1.19.0
```

# Dataset
FSE-1000 : [download here](https://drive.google.com/file/d/1YRZiJMCvGekrsEB_emVOy-hiX4yuHuZs/view?usp=sharing).
SBD-5i (For training) : [download here](https://drive.google.com/file/d/1YRZiJMCvGekrsEB_emVOy-hiX4yuHuZs/view?usp=sharing).
SBD-5i (For evaluation) : [download here](https://drive.google.com/file/d/1YRZiJMCvGekrsEB_emVOy-hiX4yuHuZs/view?usp=sharing).

# Running Code
```
#For FSE-1000 

python FSE-1000/train.py                              
                                    
#For SSD

python tools/train_SSD.py           --gpu-ids {GPU device number}
                                    --work_dir {dir to save logs and models}
                                    --config {train config file path}     
```

# Acknowledgement
Our code is based on the implementations of [Multiple Instance Active Learning for Object Detection](https://github.com/yuantn/MI-AOD).
