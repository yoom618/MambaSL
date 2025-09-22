import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding

class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout, seq_len=configs.seq_len)

        self.mamba = Mamba(
            d_model = configs.d_model,
            d_state = configs.d_ff,
            d_conv = configs.d_conv,
            expand = configs.expand,
        )

        if self.task_name == 'classification': # add classification task code. refer to other models given in tslib
            self.act = F.silu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class, bias=False)          
        elif self.task_name in ['short_term_forecast', 'long_term_forecast']:
            self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)
        else:
            raise ValueError(f"Task {self.task_name} not supported")

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def classification(self, x_enc, x_mark_enc):
        x = self.embedding(x_enc, None)
        x = self.mamba(x)

        # Output from Mamba (refer to other model codes)
        output = self.act(x)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

        return x_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            x_out = self.classification(x_enc, x_mark_enc)
            return x_out

        # other tasks not implemented