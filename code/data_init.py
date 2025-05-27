import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Токены
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

PAD_IDX = None
BOS_IDX = None
EOS_IDX = None
UNK_IDX = None



class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        assert pred.size(0) == target.size(0)
        device = pred.device

        with torch.no_grad():
            true_dist = torch.full_like(pred, self.label_smoothing / (self.tgt_vocab_size - 2))
            mask = (target != self.ignore_index)
            target_clean = target.clone()
            target_clean[~mask] = 0
            true_dist.scatter_(1, target_clean.unsqueeze(1), 1.0 - self.label_smoothing)
            true_dist[~mask] = 0

        log_probs = F.log_softmax(pred, dim=1)
        loss = -(true_dist * log_probs).sum(dim=1)
        loss = loss[mask]

        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=device)


def update_special_token_indices(word_field):
    global PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX
    PAD_IDX = word_field.vocab.stoi[PAD_TOKEN]
    BOS_IDX = word_field.vocab.stoi[BOS_TOKEN]
    EOS_IDX = word_field.vocab.stoi[EOS_TOKEN]
    UNK_IDX = word_field.vocab.stoi[UNK_TOKEN]

def tokens_to_words(tokens, vocab):
    return [vocab.itos[token] for token in tokens]


def words_to_tokens(words, vocab):
    return [vocab.stoi.get(word, vocab.stoi[UNK_TOKEN]) for word in words]