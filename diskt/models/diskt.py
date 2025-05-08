import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class DisKT(nn.Module):
    def __init__(self, num_skills, num_questions, seq_len, embedding_size, num_blocks, dropout, kq_same, 
                d_ff=256, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qr=False, l2=1e-5):
        super().__init__()
        """
        Input:
            embedding_size: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.num_skills = num_skills
        self.dropout = dropout
        self.kq_same = kq_same
        self.num_questions = num_questions
        self.l2 = l2
        self.separate_qr = separate_qr
        embed_l = embedding_size
        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions+1, embed_l) # 题目难度
            self.q_embed_diff = nn.Embedding(self.num_skills+1, embed_l) # question emb, 总结了包含当前question（concept）的problems（questions）的变化
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l) # interaction emb, 同上
        # num_skills+1 ,embedding_size
        self.q_embed = nn.Embedding(self.num_skills, embed_l)
        if self.separate_qr: 
                self.qa_embed = nn.Embedding(2*self.num_skills+1, embed_l)
        else: # false default
            self.qa_embed = nn.Embedding(3, embed_l)
        # Architecture Object. It contains stack of attention block
        self.model = Architecture(num_skills=num_skills, num_blocks=num_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    embedding_size=embedding_size, d_feature=embedding_size / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, seq_len=seq_len)

        self.ffn = FeedForward(d_model=embed_l, inner_size=embed_l*2, dropout=dropout)
        self.dual_attention = DualAttention(n_heads=num_attn_heads, d_model=embedding_size)
        self.out = nn.Sequential(
            nn.Linear(embedding_size + embed_l*2,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.loss_fn = nn.BCELoss(reduction="mean")

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_questions+1 and self.num_questions > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  embedding_size# c_ct
        if self.separate_qr:
            qa_data = q_data + self.num_skills * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, embedding_size # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
            
        return q_embed_data, qa_embed_data


    def rasch_emb(self, q_data, pid_data, target):
        # Batch First
        q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.num_questions > 0: # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct 总结了包含当前question（concept）的problems（questions）的变化
            pid_embed_data = self.difficult_param(pid_data)  # uq 当前problem的难度
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(target)
            qa_embed_data= qa_embed_data + pid_embed_data * qa_embed_diff_data


        else:
            pid_embed_data = q_embed_data

        return q_embed_data, qa_embed_data, pid_embed_data

    def forward(self, feed_dict):
        q_seq = feed_dict['questions']
        s_seq = feed_dict['skills']
        r_seq = feed_dict['responses']
        counter_attention_mask, attention_mask = feed_dict['attention_mask']
        
        masked_r = r_seq * (r_seq > -1).long()

        pos_q_embed_data, pos_qa_embed_data, _ = self.rasch_emb(masked_r * s_seq, masked_r * q_seq, 2 - masked_r)
        neg_q_embed_data, neg_qa_embed_data, _ = self.rasch_emb((1-masked_r) * s_seq, (1-masked_r) * q_seq, 2 * masked_r)
        q_embed_data, qa_embed_data, pid_embed_data = self.rasch_emb(s_seq, q_seq, masked_r)

        y1, y2, y = pos_qa_embed_data, neg_qa_embed_data, qa_embed_data
        x = q_embed_data

        distance = F.pairwise_distance(y1.view(y1.size(0), -1), y2.view(y2.size(0), -1))
        reg_loss = torch.mean(distance) * 0.001

        x = self.model(x, y)
        
        y1, y2 = self.ffn(y1), self.ffn(y2)

        y1, y2 = self.dual_attention(x, x, y1, y2, counter_attention_mask)

        x = x - (y1 + y2)
        x = x - pid_embed_data
        x = torch.cat([x, q_embed_data], dim=-1)
        x = torch.cat([x, y1 - y2], dim=-1)
        output = self.out(x).squeeze(-1)
        m1 = nn.Sigmoid()
        preds = m1(output)
        if self.training:
            out_dict = {
                "pred": preds[:, 1:],
                "true": r_seq[:, 1:].float(),
                "reg_loss" : reg_loss,
            }
        else:
            out_dict = {
                "pred": preds[:, 1:],
                "true": r_seq[:, 1:].float(),
            }
        return out_dict


    def loss(self, feed_dict, out_dict):
        pred = out_dict["pred"]
        pred_flat = pred.flatten()
        true = out_dict["true"]
        true_flat = pred.flatten()
        mask = true > -1
        mask_cf, mask = feed_dict['attention_mask']
        mask_cf, mask = mask_cf[:, 1:].bool(), mask[:, 1:].bool()
        pred_flat = torch.masked_select(pred, mask)
        true_flat = torch.masked_select(true, mask)
        loss = self.loss_fn(pred_flat, true_flat)
        
        reg_loss = out_dict["reg_loss"]
        loss = loss + reg_loss
        return loss, len(pred[mask]), true[mask].sum().item()


class DualAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(DualAttention, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_feature = d_model // n_heads
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, q, k, v1, v2, counter_attention_mask):
        batch_size = q.size(0)
        src_mask = create_mask(q, 0)
        q = q.view(batch_size, -1, self.n_heads, self.n_feature)
        k = k.view(batch_size, -1, self.n_heads, self.n_feature)
        v1 = v1.view(batch_size, -1, self.n_heads, self.n_feature)
        v2 = v2.view(batch_size, -1, self.n_heads, self.n_feature)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v1 = v1.transpose(1, 2)
        v2 = v2.transpose(1, 2)
        output_v1, output_v2, attn_weight = contradictory_attention(q, k, v1, v2, src_mask, self.dropout, counter_attention_mask)
        output_v1 = output_v1.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_v2 = output_v2.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return output_v1, output_v2

def contradictory_attention(query, key, value1, value2, mask=None, dropout=None, counter_attention_mask=None):
    "Compute 'Scaled Dot Product Attention'"
    bs, head, seqlen, d_k = query.size(0), query.size(1), query.size(2), query.size(-1)
    device = query.device
    
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e32)
    
    p_attn = F.softmax(scores, dim=-1)  # [batch_size, head, seq_len, seq_len]
    
    counter_attention_mask = counter_attention_mask.unsqueeze(1)
    counter_attention_mask = counter_attention_mask.expand(-1, head*seqlen, -1)
    counter_attention_mask = counter_attention_mask.reshape(-1, seqlen)
    
    attn_reshape = p_attn.reshape(bs*head*seqlen, -1)
    
    masked_attn = torch.where((counter_attention_mask == 1), 
                             torch.zeros_like(attn_reshape), 
                             attn_reshape)
    
    masked_attn = masked_attn.reshape(bs, head, seqlen, -1)
    
    row_sums = masked_attn.sum(dim=-1, keepdim=True)
    
    valid_rows = (row_sums > 0).float()
    safe_row_sums = row_sums + (1 - valid_rows)
    
    normalized_attn = masked_attn / safe_row_sums
    
    pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
    normalized_attn = torch.cat([pad_zero, normalized_attn[:, :, 1:, :]], dim=2)
    
    if dropout is not None:
        normalized_attn = dropout(normalized_attn)
    
    output_v1 = torch.matmul(normalized_attn, value1)
    output_v2 = torch.matmul(normalized_attn, value2)
    
    return output_v1, output_v2, normalized_attn


def create_mask(input, mask):
    seqlen = input.size(1)
    device = input.device
    nopeek_mask = np.triu(
    np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
    src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
    return src_mask


class Architecture(nn.Module):
    def __init__(self, num_skills,  num_blocks, embedding_size, d_feature,
                 d_ff, n_heads, dropout, kq_same, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            embedding_size : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = embedding_size
        """
        self.embedding_size = embedding_size

        self.blocks_2 = nn.ModuleList([
            TransformerLayer(embedding_size=embedding_size, d_feature=embedding_size // n_heads,
                                d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(num_blocks)
        ])
        self.position_emb = CosinePositionalEmbedding(embedding_size=self.embedding_size, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed


        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) # True: +FFN+残差+laynorm 非第一层与0~t-1的的q的attention, 对应图中Knowledge Retriever
            # mask=0，不能看到当前的response, 在Knowledge Retrever的value全为0，因此，实现了第一题只有question信息，无qa信息的目的
            # print(x[0,0,:])
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embedding_size, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            embedding_size, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_size, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, embedding_size)

        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        device = query.device
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True) # 只能看到之前的信息，当前的信息也看不到，此时会把第一行score全置0，表示第一道题看不到历史的interaction信息，第一题attn之后，对应value全0
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) # 残差1
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) # 残差
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.embedding_size = embedding_size
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.k_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=bias)

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

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * embedding_size

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.embedding_size)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """
    This is called by Multi-head atention object to find the values.
    """
    # d_k: 每一个头的dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    device = q.device

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # print(f"before zero pad scores: {scores.shape}")
    # print(zero_pad)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) # 第一行score置0
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, embedding_size)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, embedding_size, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() *
                             -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)
    


# https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp