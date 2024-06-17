import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
from net.res34unet import resunet
from sklearn.metrics import accuracy_score
from net import loss
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
# from apex import amp
from tensorboardX import SummaryWriter
from dataset.my_datasets import MyDataSet_seg, MyValDataSet_seg
from torch.utils import data
import datetime
import torch.nn.functional as F


INPUT_SIZE = '512, 512'
TRAIN_NUM = 485
BATCH_SIZE = 10
EPOCH = 200
LEARNING_RATE = 0.0001
fold = 0


w, h = map(int, INPUT_SIZE.split(','))
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 2
STEPS = (TRAIN_NUM/BATCH_SIZE)*EPOCH
FP16 = False
NAME = 'BUSI_fold{}/'.format(str(fold))
DB_val_loss = loss.Fusin_Dice_bce()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def val_mode_seg(valloader, model, path, epoch):
    LD = []
    LB = []
    L_totall = []
    dice = []
    sen = []
    spe = []
    acc = []
    jac_score = []
    for index, batch in enumerate(valloader):

        data, mask, name = batch
        data = data.cuda()
        mask_loss = mask.cuda()
        # print(data.max(), data.min(), mask.max(), mask.min())

        mask = mask[0].data.numpy()
        val_mask = np.int64(mask > 0)
        # print(name)

        model.eval()
        with torch.no_grad():
            # pred_loss, _, _, _ = model(data)
            pred_loss = model(data)
            pred_loss = pred_loss[0]
            # print(pred.shape)

        loss_D, loss_B = DB_val_loss(pred_loss, mask_loss)
        loss_totall = loss_D + loss_B
        LD.append(loss_D.cpu().data.numpy())
        LB.append(loss_B.cpu().data.numpy())
        L_totall.append(loss_totall.cpu().data.numpy())

        pred = torch.softmax(pred_loss, dim=1).cpu().data.numpy()
        pred_arg = np.argmax(pred[0], axis=0)

        # y_pred
        y_true_f = val_mask.reshape(val_mask.shape[0]*val_mask.shape[1], order='F')
        y_pred_f = pred_arg.reshape(pred_arg.shape[0]*pred_arg.shape[1], order='F')
        # print(y_pred_f.shape, type(y_pred_f))

        intersection = np.float64(np.sum(y_true_f * y_pred_f))
        dice.append((2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f)))
        sen.append(intersection / np.sum(y_true_f))
        intersection0 = np.float64(np.sum((1 - y_true_f) * (1 - y_pred_f)))
        spe.append(intersection0 / np.sum(1 - y_true_f))
        acc.append(accuracy_score(y_true_f, y_pred_f))
        jac_score.append(intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection))

    return np.array(acc), np.array(dice), np.array(sen), np.array(spe), np.array(jac_score), np.array(LD), np.array(LB), np.array(L_totall)


def Jaccard(pred_arg, mask):
    pred_arg = np.argmax(pred_arg.cpu().data.numpy(), axis=1)
    mask = mask.cpu().data.numpy()

    y_true_f = mask.reshape(mask.shape[0] * mask.shape[1] * mask.shape[2], order='F')
    y_pred_f = pred_arg.reshape(pred_arg.shape[0] * pred_arg.shape[1] * pred_arg.shape[2], order='F')

    intersection = np.float64(np.sum(y_true_f * y_pred_f))
    jac_score = intersection / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection)

    return jac_score


def main():
    """Create the network and start the training."""
    writer = SummaryWriter('models/' + NAME + 'log/')

    model = resunet(n_classes=2, sample=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.cuda()

    model.train()
    model.float()

    DB_loss = loss.Fusin_Dice_bce()

    # cudnn.benchmark = True

    # ############ Load training and validation data
    data_train_root = '../data/BUSI/'
    data_train_list = './dataset/BUSI_list/Training_seg_{}fold.txt'.format(str(fold))
    trainloader = data.DataLoader(MyDataSet_seg(data_train_root, data_train_list, crop_size=(w, h)),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    data_val_root = '../data/BUSI/'
    data_val_list = './dataset/BUSI_list/Validation_seg_{}fold.txt'.format(str(fold))
    valloader = data.DataLoader(MyValDataSet_seg(data_val_root, data_val_list, crop_size=(w, h)), batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    path = 'models/' + NAME
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = path + 'output.txt'

    val_jac = []
    best_jac = 0.0

    # ############ Start the training
    for epoch in range(EPOCH):
        stime = datetime.datetime.now()

        # if epoch == 2:
        #     print('epoch_debug!')

        train_loss_D = []
        train_loss_B = []
        train_loss_total = []
        train_jac = []
        loss_d = 0
        loss_b = 0
        loss_total = 0

        for i_iter, batch in tqdm(enumerate(trainloader)):

            step = (TRAIN_NUM/BATCH_SIZE)*epoch+i_iter

            images, labels, name = batch
            images = images.cuda()
            labels = labels.cuda().squeeze(1)

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, step)

            model.train()
            preds = model(images)
            # print(preds.shape)

            term = 0
            loss_D = 0
            loss_B = 0

            # if deep supervison
            labels = torch.cat([labels, labels], dim=0)
            aux_weight = [1, 0.25, 0.25, 0.25]
            # aux_weight = [1, 0.1, 0.1, 0.1]
            for nl in range(len(preds)):
                if nl == 0:
                    tem_lossD, tem_lossB = DB_loss(preds[nl], labels)
                else:
                    for nm in range(4):
                        tem_pred = preds[nl][:, nm, :, :, :]
                        tem_pred = F.interpolate(tem_pred, size=preds[0].shape[-2:], mode='bilinear', align_corners=True)
                        tem_lossD1, tem_lossB1 = DB_loss(tem_pred, labels)
                        tem_lossD += tem_lossD1
                        tem_lossB += tem_lossB1
                    tem_lossD /= 4.0
                    tem_lossB /= 4.0

                loss_D += tem_lossD * aux_weight[nl]
                loss_B += tem_lossB * aux_weight[nl]
                term += (tem_lossD + tem_lossB) * aux_weight[nl]

            # # else
            # labels = torch.cat([labels, labels], dim=0)
            # loss_D, loss_B = DB_loss(preds, labels)
            # term = loss_D + loss_B

            loss_d += loss_D.cpu().data.numpy()
            loss_b += loss_B.cpu().data.numpy()
            loss_total += term.cpu().data.numpy()

            # if FP16 is True:
            #     with amp.scale_loss(term, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            term.backward()

            optimizer.step()

            writer.add_scalar('learning_rate', lr, step)
            # writer.add_scalar('loss', term.cpu().data.numpy(), step)

            train_loss_D.append(loss_D.cpu().data.numpy())
            train_loss_B.append(loss_B.cpu().data.numpy())
            train_loss_total.append(term.cpu().data.numpy())
            train_jac.append(Jaccard(preds[0], labels))

        print("train_epoch%d: lossTotal=%f, lossDice=%f, lossBCE=%f, Jaccard=%f \n" % (epoch, np.nanmean(train_loss_total), np.nanmean(train_loss_D), np.nanmean(train_loss_B), np.nanmean(train_jac)))

        writer.add_scalar('train_dice_loss', loss_d/len(trainloader), epoch)
        writer.add_scalar('train_bce_loss', loss_b/len(trainloader), epoch)
        writer.add_scalar('a_train_total_loss', loss_total/len(trainloader), epoch)

        # ############ Start the validation
        [vacc, vdice, vsen, vspe, vjac_score, vlossD, vlossB, vloss_totall] = val_mode_seg(valloader, model, path, epoch)
        line_val = "val%d: vTotalLoss=%f, vlossD=%f, vlossB=%f, vacc=%f, vdice=%f, vse=%f, vsp=%f, vjac=%f   \n" % \
                   (epoch, np.nanmean(vloss_totall), np.nanmean(vlossD), np.nanmean(vlossB), np.nanmean(vacc),
                    np.nanmean(vdice), np.nanmean(vsen), np.nanmean(vspe), np.nanmean(vjac_score))

        print(line_val)
        f = open(f_path, "a")
        f.write(line_val)

        # ############ Plot val curve
        val_jac.append(np.nanmean(vjac_score))
        plt.figure()
        plt.plot(val_jac, label='val jaccard', color='blue', linestyle='--')
        plt.legend(loc='best')

        plt.savefig(os.path.join(path, 'jaccard.png'))
        plt.clf()
        plt.close()
        plt.show()

        plt.close('all')

        writer.add_scalar('a_val_total_loss', np.nanmean(vloss_totall), epoch)
        writer.add_scalar('val_dice_loss', np.nanmean(vlossD), epoch)
        writer.add_scalar('val_bce_loss', np.nanmean(vlossB), epoch)
        writer.add_scalar('val_Jaccard', np.nanmean(vjac_score), epoch)

        # ############ Save network
        cur_jac = np.nanmean(vjac_score)
        if cur_jac >= best_jac and epoch > 20:
            best_jac = cur_jac
            torch.save(model.state_dict(), path + 'best_epoch_' + str(epoch) + '.pth')

        etime = datetime.datetime.now()
        print('epoch_time:', etime-stime)


if __name__ == '__main__':
    main()

