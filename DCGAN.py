import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as vutils
from utils import Opt
from dataset import get_train_valid_loader
from model import netD, netG
from utils import compute_iou

def show_batch(batch):
    batch = batch.data.cpu().numpy()

    # for i in range(batch.shape[0]):
    #     img = np.squeeze(np.transpose(batch[i], (0, 2, 3, 1)))
    #     plt.figure()
    #     plt.imshow(img)
    # plt.show()

    img = np.squeeze(batch[0].transpose((1, 2, 0)))
    plt.figure()
    plt.imshow(img)
    plt.show()

def train(train_loader, netD, netG, criterion, optimizerG, optimizerD):
    for epoch in range(Opt.epochs):
        avg_lossD = 0
        avg_lossG = 0
        with open('logs.txt', 'a') as file:
            for i, sample_batched in enumerate(train_loader):
                image = sample_batched['image']
                label = sample_batched['mask']
                image = image.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)
                # Update D network
                mini_batch = label.shape[0]
                # train with real
                input = Variable(label.cuda())   # image input
                real_label = Variable(torch.ones(mini_batch).cuda())
                output = netD(input)
                D_real_loss = criterion(output, real_label)
                # train with fake
                fake = netG(Variable(image.cuda()))
                # fake = Variable((fake > 0.5).type(torch.FloatTensor).cuda())
                fake_label = Variable(torch.zeros(mini_batch).cuda())
                output = netD(fake.detach())    # detach to avoid training G on these labels
                D_fake_loss = criterion(output, fake_label)
                D_loss = D_real_loss + D_fake_loss
                netD.zero_grad()
                D_loss.backward()
                if Opt.which_pc == 0:
                    avg_lossD += D_loss.item()
                else:
                    avg_lossD += D_loss.data[0]
                optimizerD.step()
                # Update G network
                G_loss1 = criterion(fake, Variable(label.cuda()))
                output = netD(fake)
                G_loss2 = criterion(output, real_label)
                G_loss = G_loss2 + G_loss1
                if Opt.which_pc == 0:
                    avg_lossG += G_loss.item()
                else:
                    avg_lossG += G_loss.data[0]
                netG.zero_grad()
                G_loss.backward()
                optimizerG.step()

                print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                      % (epoch + 1, Opt.epochs, i + 1, len(train_loader), D_loss.data[0], G_loss.data[0]))
            avg_lossD /= i
            avg_lossG /= i
            print('epoch: ' + str(epoch+1) + ', D_loss: ' + str(avg_lossD) + ', G_loss: ' + str(avg_lossG))
            file.write('epoch: ' + str(epoch+1) + ', D_loss: ' + str(avg_lossD) + ', G_loss: ' + str(avg_lossG) + '\n')

        # save generated images
        vutils.save_image(fake.data, os.path.join(Opt.results_dir, 'img'+str(epoch)+'.png'), nrow=10, scale_each=True)

        if epoch % 50 == 0:
            if Opt.save_model:
                torch.save(netD.state_dict(), os.path.join(Opt.checkpoint_dir, 'netD-01.pt'))
                torch.save(netG.state_dict(), os.path.join(Opt.checkpoint_dir, 'netG-01.pt'))

def test(model, test_loader):
    """
    predict the masks on testing set
    :param model: trained model
    :param test_loader: test set
    :return:
        - predictions: list, for each elements, numpy array (Width, Height)
        - img_ids: list, for each elements, an image id string
    """
    predictions = []
    img_ids = []
    for batch_idx, sample_batched in enumerate(test_loader):
        data, img_id, height, width = sample_batched['image'], sample_batched['img_id'], sample_batched['height'], sample_batched['width']
        data = Variable(data.type(Opt.dtype))
        output = model.forward(data)
        # output = (output > 0.5)
        output = output.data.cpu().numpy()
        output = output.transpose((0, 2, 3, 1))    # transpose to (B,H,W,C)
        for i in range(0,output.shape[0]):
            pred_mask = np.squeeze(output[i])
            id = img_id[i]
            h = height[i]
            w = width[i]
            # in p219 the w and h above is int
            # in local the w and h above is LongTensor
            if not isinstance(h, int):
                h = h.cpu().numpy()
                w = w.cpu().numpy()
            pred_mask = resize(pred_mask, (h, w), mode='constant')
            pred_mask = (pred_mask > 0.5)
            predictions.append(pred_mask)
            img_ids.append(id)

    return predictions, img_ids

if __name__ == '__main__':
    train_loader, val_loader = get_train_valid_loader(Opt.dataset_path, batch_size=Opt.batch_size,
                                                        split=True, shuffle=Opt.shuffle,
                                                        num_workers=Opt.num_workers,
                                                        val_ratio=0.1, pin_memory=Opt.pin_memory)

    netG = netG()
    netG.load_state_dict(torch.load(os.path.join(Opt.checkpoint_dir, 'model-01.pt')))
    # netG.apply(weights_init)
    netD = netD()
    criterion = torch.nn.BCELoss()
    # netD.apply(weights_init)
    if Opt.ngpu > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
    if Opt.is_cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        criterion = criterion.cuda()

    # Optimizers
    optimizerG = torch.optim.Adam(netG.parameters(), lr=Opt.lr, betas=Opt.betas, weight_decay=Opt.weight_decay)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=Opt.lr, betas=Opt.betas, weight_decay=Opt.weight_decay)

    if Opt.is_train:
        predictions, img_ids = test(netG, val_loader)
        compute_iou(predictions, img_ids, 'UNet_IOU')
        train(train_loader, netD, netG, criterion, optimizerG, optimizerD)
        predictions, img_ids = test(netG, val_loader)
        compute_iou(predictions, img_ids, 'GAN_IOU')
    else:
        predictions, img_ids = test(netG, val_loader)
        compute_iou(predictions, img_ids, 'UNet_IOU')