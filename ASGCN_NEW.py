# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
# from layers import ListModule
from torch.autograd import Variable
from capsule import Capsule
from attentionlayer import Attn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(
            in_features, out_features))  # (hd,hd)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        # text (bs,sl+cl, hd) , hidden(bs,sl+cl, hd)
        # all_output = torch.zeros((len(text),len(self.weight),text[-1].size(0),text[-1].size(1),text[-1].size(2)))
        # for j in range(len(text)):
        all_weights = []
        # denom = torch.sum(adj, dim=2, keepdim=True) + 1
        # for i in range(len(self.weight)):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1  # degree of the i-th token
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            # all_output[j][i] = F.relu(output + self.bias[i])
            # all_weights.append(F.relu(output + self.bias[i]))
            return output + self.bias
        else:
            # all_output[j][i] = F.relu(output)
            # all_weights.append(F.relu(output))
            # all_weights.append(F.relu(output))
            return output
        # all_output.append(all_weights)
        # return all_weights

class ASGCN_NEW(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN_NEW, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float))
        self.hid_dim = opt.hidden_dim
        # self.text_lstm = nn.LSTM(opt.embed_dim, opt.hidden_dim,
        #                          num_layers=1, bidirectional=True, batch_first=True)

        self.text_lstm = DynamicLSTM(
            opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        # self.text_lstmasp = DynamicLSTM(
        #     opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        # self.gc22 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc3 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc4 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc5 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc6 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        # for i in range(opt.polarities_dim):
        #     # self.add_module('capsule_%s' % i,Capsule(opt.hidden_dim if not Config.c_bi else opt.hidden_dim * 2))
        #     self.add_module('capsule_%s' % i,Capsule( opt.hidden_dim))
        # for i in range(opt.polarities_dim):
        #     # self.add_module('capsule_%s' % i,Capsule(opt.hidden_dim if not Config.c_bi else opt.hidden_dim * 2))
        #     self.add_module('att_%s' % i, Attn(opt.hidden_dim, method='F1'))
        self.conv1 = nn.Conv1d( opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(opt.hidden_dim, opt.hidden_dim, 3, padding=1)
        # self.poool = nn.MaxPool1d(9)
        # self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        # self.fc1 = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)
        self.fc2 = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)
        # self.fc2 = nn.Linear(opt.hidden_dim*3, opt.polarities_dim)
        # self.fc4 = nn.Linear(opt.hidden_dim*4, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)
        # self.batch_nor = nn.BatchNorm1d(90)

        self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        self.layer_norm3 = torch.nn.LayerNorm(opt.hidden_dim, eps=1e-12)
        # self.layer_norm1 = torch.nn.LayerNorm(opt.hidden_dim*2, eps=1e-12)

        # self.layer_norm2 = torch.nn.LayerNorm(opt.hidden_dim*3, eps=1e-12)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len, seq_len):
        all_weights = []
        # for ii in range(len(x)):
        batch_size = x.shape[0]
        tol_len = x.shape[1]  # sl+cl
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        # concept_mod_len = concept_mod_len.cpu().numpy()

        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            # weight for text
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i, 0]):
                weight[i].append(
                    1 - (aspect_double_idx[i, 0] - j) / context_len)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i, 1] + 1, text_len[i]):
                weight[i].append(
                    1 - (j - aspect_double_idx[i, 1]) / context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
            # # weight for concept_mod
            # for j in range(seq_len, seq_len + concept_mod_len[i]):
            #     weight[i].append(1)
            # for j in range(seq_len + concept_mod_len[i], tol_len):
            #     weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        # print((weight*x).shape)
        # print((x).shape)
        # print((weight).shape)
        # all_weights.append(weight * x[ii])
        return weight * x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i, 0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i, 0], aspect_double_idx[i, 1] + 1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i, 1] + 1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask * x

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def get_state(self,bsz):
        """Get cell states and hidden states."""
        if True:
            return Variable(torch.rand(bsz, self.hid_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.hid_dim))
    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        # concept_mod_len = torch.sum(concept_mod != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat(
            [left_len.unsqueeze(1), (left_len + aspect_len - 1).unsqueeze(1)], dim=1)

        text = self.embed(text_indices)
        # text = self.batch_nor(text)
        text = self.text_embed_dropout(text)

        # textasp = self.embed(aspect_indices)
        # text = self.batch_nor(text)
        # textasp = self.text_embed_dropout(textasp)

        # text_out, _ = self.text_lstm(text)
        text_out, _ = self.text_lstm(text,text_len)
        # text_outasp, _ = self.text_lstm(textasp,aspect_len)

        # #################### SHA-RNN ########################
        # #layer_normal
        # query = text_out
        # Key = self.layer_norm(query)
        # value = self.layer_norm1(query)
        # attention_scores = torch.matmul(query, Key.transpose(-1, -2).contiguous()) \
        #                    / math.sqrt(Key.shape[-1])
        # attention_weights = F.softmax(attention_scores, dim=-1)
        # mix = torch.matmul(attention_weights, value)
        # text_out = mix+text_out
        # #################### SHA-RNN ########################

        batch_size = text_out.shape[0]
        seq_len = text_out.shape[1]
        hidden_size = text_out.shape[2] // 2
        text_out = text_out.reshape(batch_size, seq_len, hidden_size, -1).mean(dim=-1)
        # concept_mod = self.embed(concept_mod)
        # x = torch.cat([text_out, concept_mod], dim=1)
        x = text_out
        # okk = x

        x_conv = F.relu(self.conv1(
            self.position_weight(text_out, aspect_double_idx, text_len, aspect_len, seq_len).transpose(1, 2)))
        x_conv = F.relu(self.conv2(
            self.position_weight(x_conv.transpose(1, 2), aspect_double_idx, text_len, aspect_len, seq_len).transpose(1,2)))

        #gcn
        x = F.relu(self.gc1(self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len), adj))
        x2 = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len), adj))
        x3 = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len), adj))
        x4 = F.relu(self.gc4(self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len), adj))
        x5 = F.relu(self.gc5(self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len), adj))
        x6 = F.relu(self.gc6(self.position_weight(x, aspect_double_idx, text_len, aspect_len, seq_len), adj))


        x22 = self.mask(x2, aspect_double_idx)   # mask操作能将结果从0.65提升到0.72
        x2 = x22.sum(1)
        x33 = self.mask(x3, aspect_double_idx)   # mask操作能将结果从0.65提升到0.72
        x3 = x33.sum(1)
        x44 = self.mask(x4, aspect_double_idx)   # mask操作能将结果从0.65提升到0.72
        x4 = x44.sum(1)
        x55 = self.mask(x5, aspect_double_idx)   # mask操作能将结果从0.65提升到0.72
        x5 = x55.sum(1)
        x66 = self.mask(x6, aspect_double_idx)   # mask操作能将结果从0.65提升到0.72
        x6 = x66.sum(1)

        graph_mask =  x66 + x55 + x44 + x33+x22
        # graph_mask =  0.2*(x66 + x55 + x44 + x33+x22)

        hop = 10
        lambdaa = 0.01
        # lambdaa = 1
        # graph_mask = x
        for i in range(hop):
            alpha_mat = torch.matmul(graph_mask, text_out.transpose(1, 2))
            if i == hop-1:
                alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
                a1 = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x hidden_dim
            else:
                # alpha = F.softmax(alpha_mat, dim=2)
                alpha = alpha_mat
                a1 = torch.matmul(alpha, text_out).squeeze(1)  # batch_size x hidden_dim
                # graph_mask = lambdaa*F.sigmoid(a1)+graph_mask
                graph_mask = lambdaa*self.layer_norm1(F.sigmoid(a1))+graph_mask
                # graph_mask = a1+graph_mask
        # calculate hidden state attention

        text_out_mask = self.mask(text_out, aspect_double_idx)
        for i in range(hop):
            alpha_mat_text = torch.matmul(text_out_mask, text_out.transpose(1, 2))
            if i == hop-1:
                alpha_text = F.softmax(alpha_mat_text.sum(1, keepdim=True), dim=2)
                a2 = torch.matmul(alpha_text, text_out).squeeze(1)  # batch_size x hidden_dim
            else:
                # alpha_text = F.softmax(alpha_mat_text, dim=2)
                alpha_text = alpha_mat_text
                a2 = torch.matmul(alpha_text, text_out).squeeze(1)  # batch_size x hidden_dim
                # text_out_mask = lambdaa*F.sigmoid(a2)+text_out_mask
                text_out_mask = lambdaa*self.layer_norm2(F.sigmoid(a2))+text_out_mask
                # text_out_mask = a2+text_out_mask

        # # calculate CNN attention
        x_conv = self.mask(x_conv.transpose(1, 2), aspect_double_idx)
        for i in range(hop):
            alpha_mat_x_conv = torch.matmul(x_conv, text_out.transpose(1, 2))
            if i == hop-1:
                alpha_x_conv = F.softmax(alpha_mat_x_conv.sum(1, keepdim=True), dim=2)
                a3 = torch.matmul(alpha_x_conv, text_out).squeeze(1)  # batch_size x hidden_dim
            else:
                # alpha_x_conv = F.softmax(alpha_mat_x_conv, dim=2)
                alpha_x_conv = alpha_mat_x_conv
                a3 = torch.matmul(alpha_x_conv, text_out).squeeze(1)  # batch_size x hidden_dim
                # x_conv =lambdaa* F.sigmoid(a3)+x_conv
                x_conv = lambdaa*self.layer_norm3(F.sigmoid(a3))+x_conv
                # text_out_mask = a3+text_out_mask

        fnout = torch.cat((a1,a2,a3),1)


        if self.opt.use_lstm_attention:
            # output = self.fc(fnout)
            output = self.fc2(fnout)
        else:
            output = self.fc(okk)
        return output

   