import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


from layers.Embed import PositionalEmbedding, TemporalEmbedding, TimeFeatureEmbedding
from layers.MambaBlock import Mamba_TimeVariant


class TokenEmbedding_modified(nn.Module):  # original TokenEmbedding in tslib use fixed d_kernel(=3)
    def __init__(self, c_in, d_model, d_kernel=3):
        super().__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=d_kernel, padding='same', padding_mode='replicate', bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):  # original DataEmbedding in tslib use fixed max_len(=5000). This gets warning for EigenWorms dataset (seq_len=17984)
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, d_kernel=3, seq_len=5000, temporal_emb=False):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding_modified(c_in=c_in, d_model=d_model, d_kernel=d_kernel)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=max(5000, seq_len))
        if temporal_emb:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
                if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = None
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, x_mark):
        if (self.temporal_embedding is None) or (x_mark is None):
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dropout = configs.dropout
        self.save_csv = configs.save_csv  # whether to save the csv file for the outputs

        if configs.num_kernels > 0:
            if self.task_name in ['classification', 'anomaly_detection']:
                self.embedding = DataEmbedding(configs.enc_in, configs.d_model,
                                            configs.embed, configs.freq, configs.dropout, 
                                            configs.num_kernels, configs.seq_len, False)
            else:
                self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.seq_len,
                                               configs.embed, configs.freq, configs.dropout, 
                                               configs.num_kernels, configs.seq_len, True)
        else:
            self.embedding = None
        
        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model = configs.d_model,
                d_input = configs.enc_in if self.embedding is None else configs.d_model,
                d_state = configs.d_ff,
                d_conv = configs.d_conv,
                expand = configs.expand,
                timevariant_dt = bool(configs.tv_dt),    # only available in Mamba_TimeVariant
                timevariant_B = bool(configs.tv_B),      # only available in Mamba_TimeVariant
                timevariant_C = bool(configs.tv_C),      # only available in Mamba_TimeVariant
                use_D = bool(configs.use_D),            # use D(skip connection) or not
                device = configs.device,
            ),
            nn.LayerNorm(configs.d_model),
            nn.SiLU(),  # same activation ftn as Mamba Block
        )
        
        if self.task_name in ['classification']:  # one class per one sequence sample
            self.projection_type = configs.mamba_projection_type
            
            if self.projection_type == 'full':
                self.out_layer = nn.Sequential(
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_model * configs.seq_len, configs.num_class, bias=False)
                )
                nn.init.xavier_uniform_(self.out_layer[1].weight)
                
            elif self.projection_type == 'gating':
                self.out_layer = nn.Sequential(
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_model, configs.num_class, bias=False)
                )
                nn.init.xavier_uniform_(self.out_layer[1].weight)
                
                self.attn_weight = nn.Sequential(
                    nn.Linear(configs.d_model, configs.n_heads, bias=True),
                    nn.AdaptiveMaxPool1d(1),
                    nn.Softmax(dim=1),
                )
                for m in self.attn_weight.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)  # initialize to zero
                        if m.bias is not None: m.bias.data.fill_(1.0)
                
                self.gating_values = list()
            
            else: # if self.projection_type in ['last', 'average', 'max']:
                self.out_layer = nn.Sequential(
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_model, configs.num_class, bias=False)
                )
                nn.init.xavier_uniform_(self.out_layer[1].weight)
                
            
        else:
            raise ValueError(f"task_name: {configs.task_name} is not valid.")



    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['classification']:
            if self.embedding is not None:
                mamba_in = self.embedding(x_enc, None)  # (B, L_in, D)
            else:
                mamba_in = x_enc  # (B, L_in, C_in)
            
            mamba_out = self.mamba(mamba_in)  # (B, L_in, D)

            ## 1) flatten -> fully connected layer
            if self.projection_type == 'full':
                out = mamba_out.view(mamba_out.size(0), -1)  # (B, L_in, D) -> (B, L_in * D)
                out = self.out_layer(out)  # (B, L_in * D) -> (B, C_out)

            ### 2) use the last hidden state to make the final prediction
            elif self.projection_type == 'last':
                out = mamba_out[:, -1, :]  # (B, D)
                out = self.out_layer(out)  # (B, D) -> (B, C_out)
            
            ### 3) use the average of the per-time logits to make the final prediction
            elif self.projection_type == 'avg':
                out = self.out_layer(mamba_out)  # (B, L_in, D) -> (B, L_in, C_out)
                out = out * x_mark_enc.unsqueeze(2)  # Mask out the padded sequence for variable length data (e.g. JapaneseVowels)
                out = out.mean(1)  # (B, C_out)
            
            ### 4) use the maximum of the per-time logits to make the final prediction
            elif self.projection_type == 'max':
                out = self.out_layer(mamba_out)  # (B, L_in, D) -> (B, L_in, C_out)
                out = out * x_mark_enc.unsqueeze(2)  # Mask out the padded sequence for variable length data (e.g. JapaneseVowels)
                out = torch.max(out, 1)[0]  # (B, L_in, C_out) -> (B, C_out)
            
            ### 5) [proposed] use the gating value to make the final prediction
            elif self.projection_type == 'gating':
                logit_out = self.out_layer(mamba_out)  # (B, L_in, D) -> (B, L_in, C_out)
                logit_out *= x_mark_enc.unsqueeze(2)  # (B, L_in, C_out)  # Mask out the padded sequence for variable length data (e.g. JapaneseVowels)
            
                ### Compute attention weights for weighted sum of logit_out
                w_out = self.attn_weight(mamba_out)  # (B, L_in, D) -> (B, L_in, n_head) -> (B, L_in, 1)

                ### calculate the weighted average of the hidden states to make the final prediction
                out = logit_out * w_out  # (B, L_in, C_out)
                out = out.sum(1)  # (B, C_out)

                ### log w_out distribution by computing the Gini index
                gini = w_out.detach().clone().squeeze().pow(2).sum(-1, keepdim=False).tolist()  # (B, L_in) -> (B, )
                if isinstance(gini, float): gini = [gini]
                self.gating_values.extend(gini)  # (B, ) -> list of scalars

            if not self.training and self.save_csv:
                remainders = namedtuple('remainders', ['mamba_out', 'logit_out', 'w_out'])
                return out, remainders(mamba_out.detach().clone(), 
                                       logit_out.detach().clone() if self.projection_type in ['gating'] else None,
                                       w_out.detach().clone() if self.projection_type in ['gating'] else None)
            else:
                return out

        
        else:
            raise ValueError(f"task_name: {self.task_name} is not valid.")