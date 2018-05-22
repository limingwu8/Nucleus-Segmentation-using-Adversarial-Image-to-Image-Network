import os
from PIL import Image
import numpy as np
import torch
import pandas as pd

class Opt(object):
    """Configuration for training on Kaggle Data Science Bowl 2018
        Derived from the base Config class and overrides specific values
        """
    name = "DSB2018"
    which_pc = 1    # 0: train on civs linux, 1: train on p219

    # root dir of training and validation set
    root_dir = '/home/liming/Documents/dataset/dataScienceBowl2018/combined' if which_pc==0 else '/home/PNW/wu1114/Documents/dataset/dataScienceBowl2018/combined'

    # root dir of testing set
    test_dir = '/home/liming/Documents/dataset/dataScienceBowl2018/testing_data' if which_pc==0 else '/home/PNW/wu1114/Documents/dataset/dataScienceBowl2018/testing_data'

    # save segmenting results (prediction masks) to this folder
    results_dir = './images/'

    shuffle = True  # shuffle the data set
    batch_size = 256  # GTX1060 3G Memory
    epochs = 1000  # number of epochs to train
    is_train = True  # True for training, False for testing
    save_model = True  # True for saving the model, False for not saving the model

    is_cuda = torch.cuda.is_available()  # True --> GPU
    checkpoint_dir = "./checkpoint"  # dir to save checkpoints
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type

    ngpu = 2
    imageSize = 128
    nc = 1
    lr = 0.001
    weight_decay = 0.00001
    betas = (0.9, 0.999)
    save_model = 1
    dataset_path = '/home/liming/Documents/dataset/dataScienceBowl2018/combined/' if which_pc==0 else '/home/PNW/wu1114/Documents/dataset/dataScienceBowl2018/combined/'
    results_dir = './images/'
    checkpoint_dir = './checkpoint/'
    num_workers = 1     	# number of threads for data loading
    pin_memory = True   	# use pinned (page-locked) memory. when using CUDA, set to True
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


def compute_iou(predictions, img_ids, file_name):
    """
    compute IOU between two combined masks, this does not follow kaggle's evaluation
    :return: IOU, between 0 and 1
    """
    ious = []
    for i in range(0, len(img_ids)):
        pred = predictions[i]
        img_id = img_ids[i]
        mask_path = os.path.join(Opt.root_dir, img_id, 'mask.png')
        mask = np.asarray(Image.open(mask_path).convert('L'), dtype=np.bool)
        union = np.sum(np.logical_or(mask, pred))
        intersection = np.sum(np.logical_and(mask, pred))
        iou = intersection/union
        ious.append(iou)
    df = pd.DataFrame({'img_id':img_ids,'iou':ious})
    df.to_csv(file_name, index=False)