import torch
import torch.nn as nn
import torch.nn.functional as F
from data_init import DEVICE


class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.GRU(embedding.embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
        self.reduce_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.reduce_hidden(hidden).unsqueeze(0)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        energy = torch.bmm(v, energy).squeeze(1)

        attn_weights = F.softmax(energy, dim=1)
        return attn_weights


class Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, output_size):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.GRU(embedding.embedding_dim + hidden_size * 2, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 3, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)

        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(1)
        context = context.squeeze(1)
        prediction = self.fc_out(torch.cat((output, context), dim=1))

        return prediction, hidden, attn_weights.squeeze(1)


class EncoderDecoder(nn.Module):
    def __init__(self, embedding, hidden_size, output_size):
        super().__init__()
        self.embedding = embedding
        self.encoder = Encoder(embedding, hidden_size)
        self.decoder = Decoder(embedding, hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(DEVICE)
        attentions = torch.zeros(batch_size, trg_len, src.size(1)).to(DEVICE)

        encoder_outputs, hidden = self.encoder(src)

        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = output
            attentions[:, t] = attn_weights
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs, attentions

    def generate(self, src, max_len=30, bos_idx=2):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            encoder_outputs, hidden = self.encoder(src)

            input_token = torch.tensor([bos_idx] * batch_size, device=DEVICE)

            generated_tokens = [input_token]
            attentions_all = []

            for _ in range(max_len):
                output, hidden, attn_weights = self.decoder(input_token, hidden, encoder_outputs)
                top1 = output.argmax(1)
                generated_tokens.append(top1)
                attentions_all.append(attn_weights)
                input_token = top1

            generated_tokens = torch.stack(generated_tokens, dim=1)
            attentions_all = torch.stack(attentions_all, dim=1)

        return generated_tokens, attentions_all
