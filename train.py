import torch
import yaml
from matplotlib import pyplot as plt
import numpy as np

import transforms
from data import BSDS_500, NYUD, PASCAL_Context, PASCAL_VOC12
import model
# import modelRes
import cv2
# import adabound
import time
import os
import test
import test1
from datetime import datetime

if __name__ == '__main__':
    # load configures
    file_id = open('./cfgs.yaml')
    cfgs = yaml.load(file_id)
    file_id.close()

    trans = transforms.Compose([
        # transforms.RandomScale((0.7, 1.3)),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(320, scale=(0.6, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # dataset = PASCAL_Context(root=cfgs['dataset'], flag='train', transform=trans)
    # dataset = BSDS_500(root=cfgs['dataset'], VOC=True, transform=trans)
    # dataset = PASCAL_VOC12(root=cfgs['dataset'], transform=trans)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=4)

    dataset = BSDS_500(root=cfgs['dataset'], VOC=True, transform=trans)
    # dataset = NYUD(root=cfgs['dataset'], flag='train', rgb=False, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfgs['batch_size'], shuffle=True, num_workers=4)
    # saveep = cfgs['batch_size']

    # model
    for h in range(1):
        if h == 0:
            net = model.DRNet().train()
            criterion = model.Cross_Entropy1()
        if h == 1:
            net = model.DRNet1(cfgs['vgg16-5stage']).train()
            criterion = model.Cross_Entropy1()
        # net = model.DRNet(cfgs).train()
        # loss
        # criterion = model.Cross_Entropy()
        # criterion = torch.nn.BCELoss()
        # criterion = model.Loss(cfgs)
        # optimal
        if cfgs['method'] == 'Adam':
            optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': criterion.parameters()}], weight_decay=cfgs['weight_decay'])
        elif cfgs['method'] == 'SGD':
            optimizer = torch.optim.SGD([{'params': net.parameters()}, {'params': criterion.parameters()}],
                                         lr=cfgs['lr'], momentum=cfgs['momentum'], weight_decay=cfgs['weight_decay'])
        # # multi_GPU
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     net = torch.nn.DataParallel(net)
        net.to(device)
        criterion.to(device)

        # train
        for epoch in range(cfgs['max_iter']):  # loop over the dataset multiple times
            model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'], decay_steps=cfgs['decay_steps'])
            running_loss = 0.0
            for i, data in enumerate(dataloader, start=0):
                # print(datetime.now())
                start_time = time.time()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                images = data['images'].to(device)
                labels = data['labels'].to(device)

                prediction = net(images)
                loss, dp, dn = criterion(prediction, labels)
                # loss = criterion(prediction, labels)

                loss.backward()
                optimizer.step()

                # print statistics
                duration = time.time() - start_time
                print_epoch = 10
                running_loss += loss.item()
                if i % print_epoch == print_epoch - 1:  # print every 2000 mini-batches
                    examples_per_sec = 10 / duration
                    sec_per_batch = float(duration)
                    # print(p1,p2,p3)
                    format_str = '%s: step [%d, %5d/%4d],lr = %e, loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (datetime.now(), epoch + 1, i + 1, len(dataloader), optimizer.param_groups[0]['lr'],
                                        running_loss / print_epoch, examples_per_sec, sec_per_batch))
                    file_handle = open('1.txt', mode='a')
                    file_handle.write(
                        format_str % (datetime.now(), epoch + 1, i + 1, len(dataloader), optimizer.param_groups[0]['lr'],
                                      running_loss / print_epoch, examples_per_sec, sec_per_batch))
                    file_handle.write('\n')
                    file_handle.close()
                    running_loss = 0.0

                # # validation
                validation_epoch = 100
                if not os.path.exists('./validation/'):
                    os.makedirs('./validation/')
                if i % validation_epoch == validation_epoch - 1:
                    prediction = net(images)
                    prediction = prediction.cpu().detach().numpy().transpose((0, 2, 3, 1))
                    for j in range(prediction.shape[0]):
                        cv2.imwrite('./validation/' + str(j) + '.png', prediction[j] * 255)
                    ax = plt.subplot(1, 2, 1)
                    data_ = dp.cpu().detach().numpy()
                    ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                    ax = plt.subplot(1, 2, 2)
                    data_ = dn.cpu().detach().numpy()
                    ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                    plt.savefig('./validation/test' + str(epoch) + '.png')
                    plt.close('all')

                # # save model\
                # if (i + 1) % (100 * saveep) == 0 or i == len(dataloader) - 1:
                #     if np.isnan(running_loss):
                #         exit()
                #     else:
                #         torch.save(net.state_dict(), './' + cfgs['save_name'])
                #

                # validation
                # validation_epoch = 60
                # if i % validation_epoch == validation_epoch - 1:
                #     prediction = net(images)
                #     prediction = prediction.cpu().detach().numpy().transpose((0, 2, 3, 1))
                #     for j in range(prediction.shape[0]):
                #         cv2.imwrite('./validation/' + str(j) + '.png', prediction[j] * 255)
                #
                #     ax = plt.subplot(1, 2, 1)
                #     data_ = dp.cpu().detach().numpy()
                #     ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                #     ax = plt.subplot(1, 2, 2)
                #     data_ = dn.cpu().detach().numpy()
                #     ax.hist(data_, bins=np.linspace(0, 1, 100, endpoint=True))
                #     plt.savefig('./validation/test' + str(epoch) + '.png')
                #     plt.close('all')

                # save\
            if h == 0:
                save_epoch = str(epoch)
                # if epoch % sava_epoch == sava_epoch - 1:
                torch.save(net.state_dict(), './' + str(h) + save_epoch + cfgs['save_name'])
            if h == 1:
                save_epoch = str(epoch)
                # if epoch % sava_epoch == sava_epoch - 1:
                torch.save(net.state_dict(), './' + str(h) + save_epoch + cfgs['save_name'])
        if h == 0:
            print('Finished 0-Training')
            test.tests()
        if h == 1:
            print('Finished 1-Training')
            test1.tests()
    # os.system('shutdown -s -f -t 59')

