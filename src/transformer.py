import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class PostionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len=100):
        """
        constructor of sinusoid encoding class
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PostionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]
        # batch_size = x.size(0)
        seq_len  = x.size(1)
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :].to(x.device)
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512] 

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=100, embedding_dim=512, d_model=512, nhead=8, dropout=0.1, num_layers=2, linear_size=1024) -> None:
        super().__init__()

        self.encoder_emb_vectors = nn.Sequential(
            nn.Linear(in_features=vocab_size, out_features=linear_size),
            nn.ReLU(),
            nn.Linear(in_features=linear_size, out_features=linear_size),
            nn.ReLU(),
            nn.Linear(in_features=linear_size, out_features=embedding_dim),
            nn.ReLU()
        )       
            
        self.encoder_pos = PostionalEncoding(d_model=d_model, max_len=max_len)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.encoder_model = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=encoder_norm)

    def forward(self, x_vectors, src_mask=None, src_key_padding_mask=None, device='cpu'):
        x_emb_vectors = self.encoder_emb_vectors(x_vectors)
        x_pos = self.encoder_pos(x_vectors)
                
        X = x_pos + x_emb_vectors
        X = X.permute(1, 0, 2)
        
        out = self.encoder_model(X, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_len=100, embedding_dim=512, d_model=512, nhead=8, dropout=0.1, num_layers=2) -> None:
        super().__init__()
        self.decoder_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.decoder_pos = PostionalEncoding(d_model=d_model, max_len=max_len)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.decoder_model = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers, norm=decoder_norm)

        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, trg, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, device='cpu'):
        # tgt – the sequence to the decoder (required).
        # memory – the sequence from the last layer of the encoder (required).
        # tgt_mask – the mask for the tgt sequence (optional).
        # memory_mask – the mask for the memory sequence (optional).
        # tgt_key_padding_mask – the mask for the tgt keys per batch (optional).
        # memory_key_padding_mask – the mask for the memory keys per batch (optional).
        
        trg_emb = self.decoder_emb(trg)
        trg_pos = self.decoder_pos(trg)

        Y = trg_emb + trg_pos
        Y = Y.permute(1, 0, 2)
               
        dec_out = self.decoder_model(Y, 
                                     memory=memory, 
                                     tgt_mask=tgt_mask, 
                                     memory_mask=memory_mask, 
                                     tgt_key_padding_mask=tgt_key_padding_mask, 
                                     memory_key_padding_mask=memory_key_padding_mask)

        linear_out = self.linear(dec_out)
        return linear_out

class LitTransformer(pl.LightningModule):
    def __init__(self, vocab_size_src, vocab_size_tgt, src_pad_idx=0, max_len=100, embedding_dim=512, d_model=512, nhead=8, dropout=0.1, num_layers=2):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)
        
        self.encoder = TransformerEncoder(vocab_size=vocab_size_src,
                                          max_len=max_len, 
                                          embedding_dim=embedding_dim, 
                                          d_model=d_model, 
                                          nhead=nhead, 
                                          dropout=dropout, 
                                          num_layers=num_layers)

        self.decoder = TransformerDecoder(vocab_size=vocab_size_tgt, 
                                          max_len=max_len, 
                                          embedding_dim=embedding_dim, 
                                          d_model=d_model, 
                                          nhead=nhead, 
                                          dropout=dropout, 
                                          num_layers=num_layers)
        
    def forward(self, src_vectors, tgt, mask=None):
        mask = mask[:src_vectors.size(0)]

        src_pad_mask = mask.to(self.device)
        memory_key_padding_mask = mask.to(self.device)
        
        if mask is None:
            src_pad_mask = self.make_pad_mask(src_vectors, self.src_pad_idx)
            memory_key_padding_mask = self.make_pad_mask(src_vectors, self.src_pad_idx)

        src_mask = torch.ones(src_vectors.size(1), src_vectors.size(1)).to(self.device)
        tgt_mask = self.look_ahead_mask(tgt.size(1), tgt.size(1))  
        memory_mask = self.look_ahead_mask(tgt.size(1), src_vectors.size(1))
        tgt_key_padding_mask = self.make_pad_mask(tgt, self.src_pad_idx)


        enc_src = self.encoder(src_vectors, 
                               src_mask,
                               src_key_padding_mask=src_pad_mask, 
                               device=self.device)
        
        output = self.decoder(tgt,
                              memory=enc_src, 
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, 
                              memory_key_padding_mask=memory_key_padding_mask,
                              device=self.device) 
    
        return {'enc_src': enc_src, 'output': output }
    
    # for self-attention masking
    def make_pad_mask(self, seq:torch.LongTensor, padding_idx:int=None) -> torch.BoolTensor:
        """ seq: [bsz, slen], which is padded, so padding_idx might be exist.     
        if True, '-inf' will be applied sebefore applied scaled-dot attention"""
        return (seq == padding_idx).to(self.device)

    # for decoder's look-ahead masking 
    def look_ahead_mask(self, tgt_len:int, src_len:int) -> torch.FloatTensor:  
        """ this will be applied before sigmoid function, so '-inf' for proper positions needed. 
        look-ahead masking is used for decoder in transformer, 
        which prevents future target label affecting past-step target labels. """
        mask = torch.triu(torch.ones(tgt_len, src_len), diagonal=1)
        mask[mask.bool()] = -float('inf')
        return mask.to(self.device)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9, capturable=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10.0, gamma=0.95)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        X_v, y = batch['input_vectors'], batch['labels']
        attention = batch.get('attention_mask', None)
                
        y_input = y[:,:-1]
        y_expected = y[:,1:].reshape(-1)
        output = self(X_v, y_input, attention)
        pred = output['output']

        pred = pred.permute(1,0,2).reshape(-1, self.vocab_size_tgt)
        loss = self.criterion(pred, y_expected)

        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X_v, y = batch['input_vectors'], batch['labels']
        attention = batch.get('attention_mask', None)
                
        y_input = y[:,:-1]
        y_expected = y[:,1:].reshape(-1)
        output = self(X_v, y_input, attention)
        pred = output['output']

        pred = pred.permute(1,0,2).reshape(-1, self.vocab_size_tgt)
        loss = self.criterion(pred, y_expected)

        self.log('val_loss', loss)
        
        return loss
        
    def test_step(self, batch, batch_idx):
        X_v, y = batch['input_vectors'], batch['labels']
        attention = batch.get('attention_mask', None)
                
        y_input = y[:,:-1]
        y_expected = y[:,1:].reshape(-1)
        output = self(X_v, y_input, attention)
        pred = output['output']

        pred = pred.permute(1,0,2).reshape(-1, self.vocab_size_tgt)
        loss = self.criterion(pred, y_expected)

        self.log('test_loss', loss)
        
        return loss