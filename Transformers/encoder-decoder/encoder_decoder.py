import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model] when batch_first=True
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.decoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
        # nn.Transformer takes care of the encoder and decoder layers
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=nlayers, 
                                          num_decoder_layers=nlayers, 
                                          dim_feedforward=d_hid, 
                                          dropout=dropout,
                                          batch_first=True)
        
        # Output projection layer to vocabulary
        self.output_projection = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        # Weight tying: share weights between decoder embedding and output projection
        self.output_projection.weight = self.decoder.weight

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.decoder(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, 
                                  src_key_padding_mask=src_padding_mask, 
                                  tgt_key_padding_mask=tgt_padding_mask, 
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Project to vocabulary
        output = self.output_projection(output)
        return output

    def encode(self, src, src_mask):
        # src: [batch_size, src_len] when batch_first=True
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        return self.transformer.encoder(src, mask=src_mask)

    def decode(self, tgt, memory, tgt_mask):
        # tgt: [batch_size, tgt_len] when batch_first=True
        tgt = self.decoder(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        return self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
