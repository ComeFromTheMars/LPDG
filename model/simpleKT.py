import torch
from sympy.physics.vector.printing import params
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class simpleKT(nn.Module):
    def __init__(self,params, d_ff=256,loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4,
                 kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=True, l2=1e-5,
                 emb_type="qid"):
        super().__init__()
        self.model_name = "simplekt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.n_question = params.problem_num
        self.dropout = 0.5
        print(self.dropout)
        self.kq_same = kq_same
        self.n_pid = params.problem_num
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.device = params.device
        self.n_blocks=params.n_blocks
        embed_l = params.d_model
        seq_len = params.max_length
        self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)
        self.q_embed_diff = nn.Embedding(self.n_pid + 1,embed_l)
        self.qa_embed_diff = nn.Embedding(2 * self.n_pid + 2, embed_l)
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_pid+1, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_pid + 3, embed_l)
            else:  # false default
                self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(n_question=self.n_pid, n_blocks=self.n_blocks, n_heads=num_attn_heads, dropout=self.dropout,
                                  d_model=params.d_model, d_feature=params.d_model / num_attn_heads, d_ff=d_ff, kq_same=self.kq_same,
                                  model_type=self.model_type, seq_len=seq_len,device=self.device)

        self.out = nn.Sequential(
            nn.Linear(params.d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        self.loss_function = nn.BCELoss()
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self,log_dict):
        q_data = log_dict['problem_seqs_tensor'].to(self.device)+1
        target = log_dict['correct_seqs_tensor'].to(self.device)
        pid_data = log_dict['problem_seqs_tensor'].to(self.device)+1
        emb_type = self.emb_type
        correct =target + (target == -1).long()
        qa_data = q_data + correct * self.n_pid

        q_embed_data=self.q_embed(q_data)
        qa_embed_data=self.qa_embed(qa_data)

        q_embed_diff_data = self.q_embed_diff(q_data)
        pid_embed_data = self.difficult_param(pid_data)
        q_embed_data = q_embed_data + pid_embed_data *q_embed_diff_data

        d_output = self.model(q_embed_data, qa_embed_data)  # 211x512
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        target = target[:, 1:]
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output[:, 1:].reshape(-1))  # logit
        mask = labels > -1
        masked_labels = labels[mask].float()
        masked_preds = m(preds[mask])
        out_dict = {'predictions': masked_preds, 'labels': masked_labels,'mask_sum': mask.sum()}
        return out_dict

    def loss(self, outdict):
        predictions = torch.squeeze(outdict['predictions'])
        labels = torch.as_tensor(outdict['labels'], dtype=torch.float).to(self.device)
        loss = self.loss_function(predictions, labels)
        return loss
    def getPreds(self, h, problem):
        q_embed_data = self.q_embed(problem)
        q_embed_diff_data = self.q_embed_diff(problem)
        pid_embed_data = self.difficult_param(problem)
        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

        concat_q = torch.cat([h, q_embed_data], dim=-1)
        output = self.out(concat_q)
        return output


    def getks(self,log_dict):
        with torch.no_grad():
            q_data = log_dict['problem_seqs_tensor'].to(self.device)
            target = log_dict['correct_seqs_tensor'].to(self.device)
            pid_data = log_dict['problem_seqs_tensor'].to(self.device)
            emb_type = self.emb_type
            correct = target + (target == -1).long()
            qa_data = q_data + correct * self.n_pid
            # Batch First
            q_embed_data = self.q_embed(q_data)
            qa_embed_data = self.qa_embed(qa_data)
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            d_output = self.model(q_embed_data, qa_embed_data)
        return d_output

class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len,device):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'simplekt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same,device=device)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        # encoder

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y,
                      apply_pos=True)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same,device):
        super().__init__()

        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same,device=device)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.device=device

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(self.device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask,
                zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))  # 残差1
        query = self.layer_norm1(query)  # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout(  # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)  # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same,device, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.device=device
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)


        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad,self.device)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad,device):
    """
    This is called by Multi-head atention object to find the values.
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)  # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output





class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)