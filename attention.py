# attention

import lightning as L
import torch.nn as nn
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def scaled_dot_product_attention(query, key, value, tgt_mask):
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    scores = temp / scale
    scores = scores.masked_fill(tgt_mask == 0, -1e9)
    softmax = F.softmax(scores, dim=-1)
    return softmax.bmm(value)
    

class AttentionHead(nn.Module):
    def __init__(self, dim_val: int, att_size: int):
        super().__init__()
        self.q = nn.Linear(dim_val, att_size)
        self.k = nn.Linear(dim_val, att_size)
        self.v = nn.Linear(dim_val, att_size)

    def forward(self, query, key, value, tgt_mask):
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value), tgt_mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_val: int, att_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.att_size = att_size
        self.heads = nn.ModuleList(
            [AttentionHead(dim_val, att_size) for _ in range(num_heads)]
        )
        
        self.linear = nn.Linear(num_heads * att_size, dim_val)
        
    def subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).to(device)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

    def forward(self, query, key, value):
        mask = self.subsequent_mask(key.shape[1])
        # print(query.shape, key.shape, value.shape, mask.shape)
        return self.linear(
            torch.cat([h(query, key, value, mask) for h in self.heads], dim=-1)
        )