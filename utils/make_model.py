import copy
import torch.nn as nn
from models.layers.multi_headed_attention import MultiHeadedAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import Embeddings
from models.model.encoder_decoder import EncoderDecoder
from models.model.encoder import Encoder
from models.model.decoder import Decoder
from models.model.generator import Generator
from models.blocks.encoder_layer import EncoderLayer
from models.blocks.decoder_layer import DecoderLayer


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
