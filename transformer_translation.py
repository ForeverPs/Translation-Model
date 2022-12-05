import tqdm
import math
import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerEncoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

# vocabulary
# P: padding, index=0; S: start, index=5; E: end, index=6
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
map_src = {v: k for k, v in src_vocab.items()}
map_tgt = {v: k for k, v in tgt_vocab.items()}

src_vocab_len = len(src_vocab)
tgt_vocab_len = len(tgt_vocab)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, src_vocab=5, tgt_vocab=7,
                 num_encoder_layer=6, num_decoder_layer=6, dropout=0.1):
        super(TransModel, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                batch_first=True, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layer)

        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                batch_first=True, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layer)
        self.proj = nn.Linear(d_model, tgt_vocab, bias=True)

        self.src_map = nn.Embedding(src_vocab, d_model)
        self.tgt_map = nn.Embedding(tgt_vocab, d_model)

        self.pe = PositionalEncoding(d_model=d_model)

    def forward(self, src_token, tgt_token):
        # generating padding mask, 0 is the padding index
        src_padding_mask = src_token == 0
        tgt_padding_mask = tgt_token == 0
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(tgt_token.shape[-1])

        src_embed = self.src_map(src_token)
        src_embed += self.pe(src_embed)

        tgt_embed = self.tgt_map(tgt_token)
        tgt_embed += self.pe(tgt_embed)

        # - src::math: `(S, N, E)`, `(N, S, E)` if batch_first.
        # - tgt::math: `(T, N, E)`, `(N, T, E)` if batch_first.
        # - src_mask::math: `(S, S)`. subsequent (casual) mask
        # - tgt_mask::math: `(T, T)`. subsequent (casual) mask
        # - memory_mask::math: `(T, S)`. cross attention mask
        # - src_key_padding_mask::math: `(N, S)`. src sequence padding mask
        # - tgt_key_padding_mask::math: `(N, T)`. target sequence padding mask
        # - memory_key_padding_mask::math: `(N, S)`. similar to src_key_padding_mask

        src_attn = self.encoder(src_embed, src_key_padding_mask=src_padding_mask)
        tgt_attn = self.decoder(tgt=tgt_embed, memory=src_attn,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=src_padding_mask,
                                memory_mask=None,
                                tgt_mask=tgt_mask)
        vocab = self.proj(tgt_attn)
        return vocab


def get_data():
    # translation task
    max_src_len = 6
    max_tgt_len = 6

    sentences = ['ich mochte ein bier', 'i want a beer']

    # convert to token id
    encoder_input = [src_vocab[n] for n in sentences[0].split()]
    decoder_input = [tgt_vocab['S']] + [tgt_vocab[n] for n in sentences[1].split()]
    target = [tgt_vocab[n] for n in sentences[1].split()] + [tgt_vocab['E']]

    # padding to the max len within each batch
    encoder_input += [src_vocab['P']] * (max_src_len - len(encoder_input))
    decoder_input += [tgt_vocab['P']] * (max_tgt_len - len(decoder_input))
    target += [tgt_vocab['P']] * (max_tgt_len - len(target))

    # convert to torch.tensor
    # only one sample
    encoder_input = torch.tensor(encoder_input).long().unsqueeze(0)
    decoder_input = torch.tensor(decoder_input).long().unsqueeze(0)
    target = torch.tensor(target).long().unsqueeze(0)

    return encoder_input, decoder_input, target


def train(epochs, lr=1e-4):
    encoder_input, decoder_input, target = get_data()
    model = TransModel(src_vocab=src_vocab_len, tgt_vocab=tgt_vocab_len)
    # ignore the padding loss
    # criterion = nn.CrossEntropyLoss(ignore_index=src_vocab['P'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = 1e9
    model.train()
    for epoch in tqdm.tqdm(range(epochs)):
        optimizer.zero_grad()
        logits = model(encoder_input, decoder_input)
        loss = criterion(logits.view(-1, logits.shape[-1]), target.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

        # pred_index = torch.argmax(logits, dim=-1).squeeze().detach().numpy().tolist()
        # pred_sentence = ' '.join([map_tgt[curr_index] for curr_index in pred_index])
        # print('Translation:', pred_sentence)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'trans_model.pth')


def infer(src_input):
    model = TransModel(src_vocab=src_vocab_len, tgt_vocab=tgt_vocab_len)
    model.load_state_dict(torch.load('trans_model.pth', map_location='cpu'), strict=True)
    model.eval()

    decoder_seq = [tgt_vocab['S']]
    with torch.no_grad():
        while decoder_seq[-1] != tgt_vocab['E']:
            decoder_input = torch.tensor(decoder_seq).unsqueeze(0)
            out = model(src_input, decoder_input).view(-1, tgt_vocab_len)
            pred_index = torch.argmax(out[-1, :])
            decoder_seq += [int(pred_index)]
    pred_sentence = ' '.join([map_tgt[curr_index] for curr_index in decoder_seq[1:-1]])
    print('Output tgt:', pred_sentence + '.')


if __name__ == '__main__':
    # training
    # train(epochs=30, lr=1e-4)

    # validation
    encoder_input, decoder_input, target = get_data()
    src_sentence = ' '.join([map_src[curr_index] \
                             for curr_index in encoder_input.numpy().tolist()[0] if curr_index != src_vocab['P']])
    print('Input src:', src_sentence + '.')
    infer(encoder_input)
