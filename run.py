import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from torch.utils.data import  DataLoader
from Dataset import Dataset_for_run,pad_batch_fn
from dataset.RegenerationDataset import RegenerationDataset
from utils import initialize_effect, get_dataset_information, get_optimal_value, get_Data_Time
from dataset.RegenerationDataset import batch_fn as pad
from tqdm import tqdm
import time
import argparse


def train_student(params):
    dataset=params.dataset
    setting_path= './dataset/' + dataset+ '/settings.json'
    dataset_information = get_dataset_information(dataset=dataset, max_length=['max_length'], path=setting_path)
    params.skill_num = int(dataset_information['skill_num'])
    params.problem_num = int(dataset_information['problem_num'])
    params.sequence_num = int(dataset_information['sequence_num'])
    params.datatime = get_Data_Time()
    batch_size = params.batch_size
    epoch_num = params.epoch_num
    effect = initialize_effect()
    train_path = './dataset/' + dataset +'/TrainSet/train_data.csv'
    train_dataset = RegenerationDataset(None, t='train', path=train_path)
    if params.labeltype=="original":
        regenerationDataset = train_dataset
    elif params.labeltype=='diffusion':
        train_regen = './dataset/' + dataset + '/wo_pooling_TrainSet_Z_3in10.0001_r.pth'
        train_regeneration=torch.load(train_regen)
        regenerationDataset=RegenerationDataset(train_regeneration,t='train_regen',path=None)
        regenerationDataset=train_dataset+regenerationDataset
    regenerationloader=DataLoader(regenerationDataset,batch_size=batch_size,drop_last=True,shuffle=True,collate_fn=pad)
    val_data = Dataset_for_run(dataset=params.dataset,mode='val')
    test_data = Dataset_for_run(dataset=params.dataset,mode='test')
    val_dataloader = DataLoader(val_data, batch_size=256, shuffle=True,drop_last=True, num_workers=2, collate_fn=pad_batch_fn)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True,drop_last=True, num_workers=2, collate_fn=pad_batch_fn)
    print('params.model: ', params.model)
    print('{0}.{0}'.format(params.model))
    model_name = eval('{0}.{0}'.format(params.model))
    model = model_name(params)
    model = model.to(params.device)
    optimzer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    for epoch in range(1, epoch_num + 1):
        epoch_train_loss = 0.0
        print('Epoch.{} starts.'.format(epoch))
        current_time = time.time()
        predictions_list = list()
        labels_list = list()
        model.train()
        for index, log_dict in enumerate(tqdm(regenerationloader)):
            optimzer.zero_grad()
            output_dict = model.forward(log_dict)
            predictions_list.append(torch.squeeze(output_dict['predictions'].detach()))
            labels_list.append(torch.squeeze(output_dict['labels'].detach()))
            loss = model.loss(output_dict)
            loss.backward()
            optimzer.step()
            epoch_train_loss += loss.item()
        # scheduler.step()
        metrics = get_metrics(predictions_list, labels_list)
        print("for train ", "loss:", epoch_train_loss, "metrics: ", metrics)
        effect['train_loss'].append(epoch_train_loss)
        effect['train_acc'].append(metrics['acc'])
        effect['train_auc'].append(metrics['auc'])
        train_str = 'For train. loss: {}, acc: {}, auc: {}'.format(str(epoch_train_loss), str(metrics['acc']),str(metrics['auc']))
 
        ################################### After one round of training, verify the model ##################################################
        # 验证实验代码
        epoch_val_loss = 0.0
        predictions_list = list()
        labels_list = list()
        model.eval()
        for index, log_dict in enumerate(val_dataloader):
            with torch.no_grad():
                output_dict = model.forward(log_dict)
                predictions_list.append(torch.squeeze(output_dict['predictions'].detach()))
                labels_list.append(torch.squeeze(output_dict['labels'].detach()))
                loss = model.loss(output_dict)
                epoch_val_loss += loss.item()
        metrics = get_metrics(predictions_list, labels_list)
        print("for val ", "loss:", epoch_val_loss, "metrics:", metrics)
        effect['val_loss'].append(epoch_val_loss)
        effect['val_acc'].append(metrics['acc'])
        effect['val_auc'].append(metrics['auc'])
        val_str = 'For val. loss: {}, acc: {}, auc: {}'.format(str(epoch_val_loss), str(metrics['acc']),str(metrics['auc']))

        ################################### After one round of training, test the model ##################################################
        predictions_list = list()
        labels_list = list()
        epoch_test_loss = 0.0
        model.eval()
        for index, log_dict in enumerate(test_dataloader):
            with torch.no_grad():
                output_dict = model.forward(log_dict)
                predictions_list.append(torch.squeeze(output_dict['predictions'].detach()))
                labels_list.append(torch.squeeze(output_dict['labels'].detach()))
                loss = model.loss(output_dict)
                epoch_test_loss += loss.item()
        metrics = get_metrics(predictions_list, labels_list)
        print("for test ", "loss:", epoch_test_loss, "metrics:", metrics)
        effect['test_loss'].append(epoch_test_loss)
        effect['test_acc'].append(metrics['acc'])
        effect['test_auc'].append(metrics['auc'])
        test_str = 'For test. loss: {}, acc: {}, auc: {}'.format(str(epoch_test_loss), str(metrics['acc']),str(metrics['auc']))
        save_result(params.labeltype,train_str, val_str, test_str, model=params.model, epoch=epoch, dataset=params.dataset,
                    datatime=params.datatime, k_fold_num=0)
        print("epoch_time: ", time.time() - current_time)
        if params.early_stop >= 2 and epoch > params.early_stop:
            val_auc_temp = effect['val_auc'][-params.early_stop:]
            max_val_auc_temp = max(val_auc_temp)
            if max_val_auc_temp == val_auc_temp[0]:
                print("epoch=", epoch, "early stop!")
                break
    optimal_effect = get_optimal_value(effect)
    for item in optimal_effect.items():
        print(item)
    return optimal_effect
#
def get_metrics(prediction_list, label_list):
    predictions = torch.squeeze(torch.cat(prediction_list).cpu())
    predictions_round = torch.round(predictions)
    labels = torch.squeeze(torch.cat(label_list).cpu())
    accuracy = accuracy_score(labels, predictions_round)
    auc = roc_auc_score(labels, predictions)
    return_dict = {'acc': float('%.4f' % accuracy), 'auc': float('%.4f' % auc)}
    return return_dict

def save_effect(effect, settings, k_fold_num):
    save_path = 'Model_Save/' + settings['model'] + '/' + settings['dataset'] + '_cross_' + \
                '_Effect_fold' + str(k_fold_num) + \
                settings['data_time']
    np.save(save_path, effect)

def save_result(t,train_str, val_str, test_str, epoch, model, dataset, datatime, k_fold_num):
    save_path = 'Result_save/' + model + '/' + dataset + t + datatime + '.txt'
    f = open(save_path, "a")
    f.write('Epoch.' + str(epoch) +t+ '\n')
    f.write(train_str + '\n')
    f.write(val_str + '\n')
    f.write(test_str + '\n')
    f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from model import AKT,FAKT,sparseKT,stableKT,simpleKT,SAKT
    parser.add_argument('--seed', type=int, default=2025, help='number of folds for cross_validation.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--dataset', type=str, default='Assistment12', help='choose a dataset.')
    parser.add_argument('--model', type=str, default='simpleKT', help='choose a model.')
    parser.add_argument('--device', type=str, default='cuda', help='choose a device.')
    parser.add_argument('--max_length', type=int, default=50, help='choose a value for max length.')
    parser.add_argument('--epoch_num', type=int, default=100, help='epoch num 100')
    parser.add_argument('--early_stop', type=int, default=10, help='number of early stop for AUC.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size.')
    parser.add_argument('--embed_size', type=int, default=64, help='embed_size.')
    parser.add_argument('--labeltype', type=str, default='diffusion', help='add data label type')
    # parser.add_argument('--labeltype', type=str, default='original', help='dataset type')

    # for SAKT simpleKT and FAKT
    parser.add_argument('--d_model', type=int, default=256,help=' embed size')
    parser.add_argument('--n_blocks', type=int, default=2, help=' n_blocks')
    parser.add_argument('--num_attn_heads', type=int, default=8, help='number of attention heads.')
    params = parser.parse_args()
    dataset = params.dataset
    train_student(params)


