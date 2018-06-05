import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
import imageio

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
    batch_size = 128  # GTX1060 3G Memory
    epochs = 1000  # number of epochs to train
    is_train = True  # True for training, False for testing
    save_model = True  # True for saving the model, False for not saving the model

    is_cuda = torch.cuda.is_available()  # True --> GPU
    checkpoint_dir = "./checkpoint"  # dir to save checkpoints
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type

    ngpu = 1
    imageSize = 128
    nc = 3
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

    save_img_id = [
        '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9',
        '0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe',
        '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e',
        '0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e',
        '0b2e702f90aee4fff2bc6e4326308d50cf04701082e718d4f831c8959fbcda93',
        '0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe',
        '0bf4b144167694b6846d584cf52c458f34f28fcae75328a2a096c8214e01c0d0',
        '0bf33d3db4282d918ec3da7112d0bf0427d4eafe74b3ee0bb419770eefe8d7d6',
        '0c2550a23b8a0f29a7575de8c61690d3c31bc897dd5ba66caec201d201a278c2',
        '0c6507d493bf79b2ba248c5cca3d14df8b67328b89efa5f4a32f97a06a88c92c',
        '0d2bf916cc8de90d02f4cd4c23ea79b227dbc45d845b4124ffea380c92d34c8c',
        '0d3640c1f1b80f24e94cc9a5f3e1d9e8db7bf6af7d4aba920265f46cadc25e37',
        '0ddd8deaf1696db68b00c600601c6a74a0502caaf274222c8367bdc31458ae7e',
        '0e4c2e2780de7ec4312f0efcd86b07c3738d21df30bb4643659962b4da5505a3',
        '0e5edb072788c7b1da8829b02a49ba25668b09f7201cf2b70b111fc3b853d14f',
        '0e21d7b3eea8cdbbed60d51d72f4f8c1974c5d76a8a3893a7d5835c85284132e'
    ]


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


def intersect_index(a, b):
    index = []
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i]==b[j]:
                index.append((i,j))
    return index

def make_gif(root, interval = 0):
    gen_gif_plots = []
    files = os.listdir(root)
    for i in range(0, len(files)-1, interval):
        file_name = 'img' + str(i) + '.png'
        gen_gif_plots.append(imageio.imread(os.path.join(root, file_name)))

    imageio.mimsave('./images/img.gif', gen_gif_plots, fps=3)


if __name__ == '__main__':
    make_gif('./images/', 10)