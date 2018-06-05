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
from utils import compute_iou, intersect_index

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
        save_img = torch.zeros(16, 1, 128, 128)
        with open('logs.txt', 'a') as file:
            for i, sample_batched in enumerate(train_loader):
                image = sample_batched['image']
                label = sample_batched['mask']
                image = image.type(torch.FloatTensor)
                label = label.type(torch.FloatTensor)
                # Update D network
                mini_batch = label.shape[0]
                # train with real
                input = image*label
                input = Variable(input.cuda())   # image input
                real_label = Variable(torch.ones(mini_batch).cuda())
                output = netD(input)
                D_real_loss = criterion(output, real_label)
                # train with fake
                fake = netG(Variable(image.cuda()))
                # fake = Variable((fake > 0.5).type(torch.FloatTensor).cuda())
                fake_concat = fake*Variable(image.cuda())
                fake_label = Variable(torch.zeros(mini_batch).cuda())
                output = netD(fake_concat.detach())    # detach to avoid training G on these labels
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
                output = netD(fake_concat)
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

                # get generated images
                idice = intersect_index(save_img_id, sample_batched['img_id'])
                for j in range(len(idice)):
                    idx = idice[j]
                    save_img[idx[0]] = fake[idx[1]].data.cpu()

            avg_lossD /= i
            avg_lossG /= i
            print('epoch: ' + str(epoch+1) + ', D_loss: ' + str(avg_lossD) + ', G_loss: ' + str(avg_lossG))
            file.write('epoch: ' + str(epoch+1) + ', D_loss: ' + str(avg_lossD) + ', G_loss: ' + str(avg_lossG) + '\n')

        # save generated images
        vutils.save_image(save_img, os.path.join(Opt.results_dir, 'img'+str(epoch)+'.png'), nrow=4, scale_each=True)

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
        # predictions, img_ids = test(netG, val_loader)
        # compute_iou(predictions, img_ids, 'UNet_IOU')
        train(train_loader, netD, netG, criterion, optimizerG, optimizerD)
        # predictions, img_ids = test(netG, val_loader)
        # compute_iou(predictions, img_ids, 'GAN_IOU')
    else:
        predictions, img_ids = test(netG, val_loader)
        compute_iou(predictions, img_ids, 'UNet_IOU')