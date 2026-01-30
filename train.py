import time
import torch
import math
import numpy as np
import os

from tqdm import tqdm
import torch.nn.functional as F
from dataloader import load_data
import gc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
from configp import get_args

from Net import NET_DUAL
from Net import HybridLoss

from datetime import datetime
from util import model_save


def train_epoch(epoch, model, train_data, fo, criterion):
    model.train()
    n_batches = len(train_data)
    learning_rate = '2'
    # 学习率设置
    if learning_rate == '1':
        if epoch < 30:
            LEARNING_RATE = 0.0001
        elif epoch < 40:
            LEARNING_RATE = 0.00003
        elif epoch < 50:
            LEARNING_RATE = 0.000001
        else:
            LEARNING_RATE = 0.0000001
    elif learning_rate == '2':
        if epoch < 10:
            LEARNING_RATE = 0.00001
        elif epoch < 20:
            LEARNING_RATE = 0.00001
        elif epoch < 30:
            LEARNING_RATE = 0.00001
        elif epoch < 40:
            LEARNING_RATE = 0.000003
        elif epoch < 50:
            LEARNING_RATE = 0.000001
        else:
            LEARNING_RATE = 0.000001


    else:
        LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': LEARNING_RATE},
    ], lr=0.0001)

    total_loss = 0
    source_correct = 0
    total_label = 0
    loss = None
    output = None

    for (mri_data, pet_data, label) in tqdm(train_data, total=n_batches):
        # data = image_reshape(data, [108, 126, 108])
        mri_data, pet_data, label = mri_data.cuda(), pet_data.cuda(), label.cuda()
        result = model(mri_data, pet_data)


        if type(result) is tuple:
            if len(result) == 3:
                P_l, P_r, output = result
                CosineLoss = 0
                HybridLoss = criterion['HybridLoss'](P_l, P_r, output, label)
                loss = HybridLoss + CosineLoss
            elif len(result) == 5:
                left, right, P_l, P_r, output = result
                cosLossLabel = -torch.ones(left.size(0), dtype=torch.long).cuda()
                CosineLoss = criterion['CosineLoss'](F.normalize(left, dim=1), F.normalize(right, dim=1),
                                                     cosLossLabel).cuda()
                HybridLoss = criterion['HybridLoss'](P_l, P_r, output, label)
                loss = HybridLoss + CosineLoss
        else:
            loss = torch.nn.functional.cross_entropy(result, label)
            output = result

        _, preds = torch.max(output, 1)
        source_correct += preds.eq(label.data.view_as(preds)).cpu().sum()
        total_label += label.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


    acc = source_correct / total_label
    mean_loss = total_loss / n_batches
    print(f'Epoch: [{epoch:2d}], '
          f'Loss: {mean_loss:.6f}, ',
          f'LR: {LEARNING_RATE:.6f}')
    log_str = 'Epoch: ' + str(epoch) \
              + ' Loss: ' + str(mean_loss) \
              + ' train_acc: ' + str(acc) + '\n'
    fo.write(log_str)
    del acc, train_data, n_batches
    gc.collect()

    return mean_loss, source_correct, total_label


def val_model(epoch, val_data, model, log_best):
    model.eval()
    print('Test a model on the val data...')
    correct = 0
    total = 0
    total_loss = 0
    true_label = []
    data_pre = []
    output = None
    loss = None
    with torch.no_grad():
        for (mri_data, pet_data, labels) in tqdm(val_data):

            mri_data, pet_data, labels = mri_data.cuda(), pet_data.cuda(), labels.cuda()
            result = model(mri_data, pet_data)


            if type(result) is tuple:
                if len(result) == 3:
                    P_l, P_r, output = result
                    CosineLoss = 0
                    HybridLoss = criterion['HybridLoss'](P_l, P_r, output, labels)
                    loss = HybridLoss + CosineLoss
                elif len(result) == 5:
                    left, right, P_l, P_r, output = result
                    cosLossLabel = -torch.ones(left.size(0), dtype=torch.long).cuda()
                    CosineLoss = criterion['CosineLoss'](F.normalize(left, dim=1), F.normalize(right, dim=1),
                                                         cosLossLabel).cuda()
                    HybridLoss = criterion['HybridLoss'](P_l, P_r, output, labels)
                    loss = HybridLoss + CosineLoss
            else:
                loss = torch.nn.functional.cross_entropy(result, labels)
                output = result

            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    mean_loss = total_loss / len(val_data)
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    log_str = 'Epoch: ' + str(epoch) \
              + '\n' \
              + 'TP: ' + str(TP) + ' TN: ' + str(TN) + ' FP: ' + str(FP) + ' FN: ' + str(FN) \
              + '  ACC:  ' + str(ACC) \
              + '  SEN:  ' + str(SEN) \
              + '  SPE:  ' + str(SPE) \
              + '  AUC:  ' + str(AUC) \
              + '\n'
    log_best.write(log_str)
    del correct, total, true_label, data_pre, mri_data, pet_data, labels, output, TN, FP, FN, TP
    gc.collect()
    return ACC, SEN, SPE, AUC, mean_loss  #, feature_for_GCN


def t_model(test_data, model):
    model.eval()
    print('Test a model on the test data...')
    correct = 0
    total = 0
    true_label = []
    data_pre = []
    with torch.no_grad():
        for (mri_data, pet_data, labels) in tqdm(test_data):
            mri_data, pet_data, labels = mri_data.cuda(), pet_data.cuda(), labels.cuda()
            output = model(mri_data, pet_data)
            if type(output) is tuple:
                output = output[-1]
            _, predicted = torch.max(output, 1)
            true_label.extend(list(labels.cpu().flatten().numpy()))
            data_pre.extend(list(predicted.cpu().flatten().numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * (TP) / (TP + FN)
    SPE = 100 * (TN) / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print('The result of test data: \n')
    print('TP:', TP, 'FP:', FP, 'FN:', FN, 'TN:', TN)
    print('ACC: %.4f %%' % ACC)
    print('SEN: %.4f %%' % SEN)
    print('SPE: %.4f %%' % SPE)
    print('AUC: %.4f %%' % AUC)
    del correct, total, true_label, data_pre, mri_data, pet_data, labels, output, TN, FP, FN, TP
    gc.collect()
    # return ACC, SEN, SPE, AUC


if __name__ == '__main__':
    experience_log = "Name_" + str(datetime.now().timestamp())
    log_path = "./" + experience_log
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_data: torch.utils.data.DataLoader = load_data(args, args.train_root_path_MRI, args.train_root_path_PET, args.AD_dir, args.CN_dir)
    test_data: torch.utils.data.DataLoader = load_data(args, args.test_root_path_MRI, args.test_root_path_PET, args.AD_dir, args.CN_dir)
    # val_data: torch.utils.data.DataLoader = load_data(args, args.val_root_path_MRI, args.val_root_path_PET,
    #                                                    args.AD_dir, args.CN_dir)

    model = NET_DUAL.NET_DUAL(2, (105, 125, 105), 25).cuda()
    # model.mri_network.load_state_dict(torch.load('./Net/parameter/NET_MRI1_PET1/MRI/ACC88.05970149253731_SEN88.88888888888889_SPE87.09677419354838_AUC87.99283154121864_epoch72.pt'))
    # model.pet_network.load_state_dict(torch.load('./Net/parameter/NET_MRI1_PET1/PET/ACC90.0_SEN83.72093023255815_SPE95.74468085106383_AUC89.732805541811_epoch33.pt'))
    # for param in model.mri_network.parameters():
    #     param.requires_grad = False
    # for param in model.pet_network.parameters():
    #     param.requires_grad = False

    criterion = {
        'HybridLoss': HybridLoss.HybridLoss().cuda(),
        'CosineLoss': torch.nn.CosineEmbeddingLoss().cuda()
    }

    loss_best_model_dir = None
    acc_best_model_dir = None
    train_best_loss = 10000
    val_best_loss = 10000
    train_best_acc = 0
    val_best_acc = 0
    t_SEN = 0
    t_SPE = 0
    t_AUC = 0
    t_precision = 0
    t_f1 = 0

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    test_acc_all = []
    count = 0

    since = time.time()

    for epoch in range(1, args.nepoch + 1):
        # training
        fo = open(log_path + "/test.txt", "a")
        log_best = open(log_path + '/log_best.txt', 'a')
        train_loss, train_correct, len_train = train_epoch(epoch, model, train_data, fo, criterion)
        # save model
        # print acc and auc
        if train_loss < train_best_loss:
            train_best_loss = train_loss
            path_dir = log_path + '/model/bestLoss/epoch' + str(epoch) + '_Loss' + str(train_best_loss) + '.pt'
            loss_best_model_dir = path_dir
            model_save(model, path_dir)
        train_acc = 100. * train_correct / len_train
        if train_acc > train_best_acc:
            max_acc = train_acc
        print('current loss: ', train_loss, 'the best loss: ', train_best_loss)
        print(f'train_correct/train_data: {train_correct}/{len_train} accuracy: {train_acc:.2f}%')

        # log result in text
        ACC, SEN, SPE, AUC, val_loss = val_model(epoch, test_data, model, log_best)
        if ACC >= val_best_acc:  # if val_loss < val_best_loss
            target_best_acc = ACC  # val_best_loss = val_loss
            val_best_acc = target_best_acc
            t_SEN = SEN
            t_SPE = SPE
            t_AUC = AUC
            path_dir = log_path + '/model/bestACC/ACC' + str(val_best_acc) + '_SEN' + str(t_SEN) + '_SPE' + str(
                t_SPE) + '_AUC' + str(t_AUC) + '_epoch' + str(epoch) + '.pt'
            acc_best_model_dir = path_dir
            model_save(model, path_dir)
        if epoch >= (args.nepoch - 10):
            path_dir = log_path + '/model/Last10/ACC' + str(ACC) + '_SEN' + str(SEN) + '_SPE' + str(SPE) + '_AUC' + str(
                AUC) + '_epoch' + str(epoch) + '.pt'
            model_save(model, path_dir)

            #np.savez('feature_' + str(count), feature_for_GCN.detach().cpu().numpy())
            #count = count+1

        log_best.write('The best result:\n')
        log_best.write('ACC:  ' + str(val_best_acc) + '  SEN:  ' + str(t_SEN) + '  SPE:  ' + str(
            t_SPE) + '  AUC:  ' + str(t_AUC) + '\n\n')

        print(f'The train acc of this epoch: {train_acc:.2f}%')
        print(f'The best acc: {train_acc:.2f}% \n')
        fo.write(
            'train_acc: ' + str(train_acc) + ' The current total loss: ' + str(train_loss) + ' The best loss: ' + str(
                train_best_loss) + '\n\n')

        # save data for img
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        test_acc_all.append(ACC)

        del train_loss, train_correct, len_train, train_acc
        gc.collect()
        fo.close()
        log_best.close()

    print(experience_log)
    # testing
    print("loss best model test:")
    model.load_state_dict(torch.load(loss_best_model_dir, weights_only=True))
    t_model(test_data, model)
    print("val acc best model test:")
    model.load_state_dict(torch.load(acc_best_model_dir, weights_only=True))
    t_model(test_data, model)

    time_use = time.time() - since
    print("Train and Test complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    train_process = pd.DataFrame(
        data={"epoch": range(args.nepoch),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "test_acc_all": test_acc_all}
    )
    train_process.to_csv(log_path + '/train_process_result.csv')

    # draw img
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all, label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    plt.plot(train_process.epoch, train_process.test_acc_all, label="Val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(log_path + "/accuracy_loss.png")
    #plt.show()
