# -*- coding: utf-8 -*-
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset



class Dataset_for_run(Dataset):
    def __init__(self,
                 dataset='Assistment12',
                 mode='train',
                 ):

        assert mode == 'train' or mode == 'test' or mode == 'val'

        if mode == 'train':
            filename = 'TrainSet'
        elif mode == 'val':
            filename = 'ValSet'
        else:
            filename = 'TestSet'

        self.file_path = './dataset/' + dataset +'/'+ str(filename) + '/' + str(mode) +'_data.csv'
        data_use = pd.read_csv(self.file_path)
        self.len = len(data_use)
        self.problem_seq = data_use['problem_seq']
        self.skill_seq = data_use['skill_seq']
        self.correct_seq = data_use['correct_seq']

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        batch_dict = {
            'problem_seq': list(map(int, self.problem_seq[item].strip('[').strip(']').split(','))),
            'skill_seq': list(map(int, self.skill_seq[item].strip('[').strip(']').split(','))),
            'correct_seq': list(map(eval, self.correct_seq[item].strip('[').strip(']').split(',')))
        }

        return batch_dict
def pad_batch_fn(many_batch_dict):
    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['problem_seq']), reverse=True)
    problem_seqs = [torch.LongTensor(seq['problem_seq']) for seq in sorted_batch]
    skill_seqs = [torch.LongTensor(seq['skill_seq']) for seq in sorted_batch]

    correct_seqs = [torch.LongTensor(seq['correct_seq']) for seq in sorted_batch]


    seqs_length = torch.LongTensor(list(map(len, skill_seqs)))

    problem_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    skill_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()

    correct_seqs_tensor = torch.full((len(sorted_batch), seqs_length.max()), -1).long()


    for idx, (problem_seq, skill_seq, correct_seq,  seq_len) in enumerate(
            zip(problem_seqs, skill_seqs, correct_seqs,  seqs_length)):
        problem_seqs_tensor[idx, :seq_len] = torch.LongTensor(problem_seq)
        skill_seqs_tensor[idx, :seq_len] = torch.LongTensor(skill_seq)

        correct_seqs_tensor[idx, :seq_len] = torch.LongTensor(correct_seq)


    return_dict = {'problem_seqs_tensor': problem_seqs_tensor,
                   'skill_seqs_tensor': skill_seqs_tensor,

                   'correct_seqs_tensor': correct_seqs_tensor,

                   'seqs_length': seqs_length}
    return return_dict





