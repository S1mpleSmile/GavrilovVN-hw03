import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from prelaunch import load_dataset, load_word_field
from model import EncoderDecoder
from data_init import DEVICE, PAD_IDX, LabelSmoothingLoss, BOS_TOKEN, EOS_TOKEN, update_special_token_indices

wandb.init(project="headline_generator", name="rusvectores_training")

HIDDEN_SIZE = 512
BATCH_SIZE = 64
N_EPOCHS = 10
CLIP = 1
BEST_ROUGE_L = 0.0  # Лучший результат

# Загрузка данных и словаря
train_data = load_dataset("../data/train")
val_data = load_dataset("../data/val")
word_field = load_word_field("../data")
update_special_token_indices(word_field)
UNK_IDX = word_field.vocab.stoi["<unk>"]

# Итераторы
train_iter, val_iter = BucketIterator.splits(
    (train_data, val_data),
    batch_size=BATCH_SIZE,
    device=DEVICE,
    sort_key=lambda x: len(x.source),
    sort_within_batch=True
)

# Предобученный эмбеддинг
embedding = nn.Embedding.from_pretrained(word_field.vocab.vectors, freeze=False, padding_idx=PAD_IDX)

model = EncoderDecoder(embedding, hidden_size=HIDDEN_SIZE, output_size=len(word_field.vocab)).to(DEVICE)

optimizer = optim.Adam(model.parameters())

criterion = LabelSmoothingLoss(label_smoothing=0.1, tgt_vocab_size=len(word_field.vocab), ignore_index=PAD_IDX)

rouge = evaluate.load("rouge")


def train():
    model.train()
    epoch_loss = 0

    with tqdm(total=len(train_iter), desc=f"Training Epoch {epoch}") as pbar:
        for i, batch in enumerate(train_iter):
            src = batch.source
            trg = batch.target

            optimizer.zero_grad()
            output, _ = model(src, trg)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss

            pbar.set_postfix(loss=step_loss, ppx=torch.exp(loss).item())
            pbar.update(1)

            wandb.log({"Batch Train Loss": step_loss}, step=epoch * len(train_iter) + i)

    return epoch_loss / len(train_iter)


def evaluate_model():
    model.eval()
    val_loss = 0
    preds = []
    refs = []

    with torch.no_grad():
        for batch in val_iter:
            src = batch.source
            trg = batch.target
            output, _ = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output_flat, trg_flat)
            val_loss += loss.item()

            generated_tokens, _ = model.generate(src, max_len=trg.shape[1], bos_idx=word_field.vocab.stoi[BOS_TOKEN])
            for i in range(src.shape[0]):
                pred_tokens = generated_tokens[i].cpu().numpy()
                pred_text = tokens_to_text(pred_tokens, word_field)
                trg_tokens = trg[i].cpu().numpy()
                trg_text = tokens_to_text(trg_tokens, word_field)
                preds.append(pred_text)
                refs.append(trg_text)

    rouge_scores = rouge.compute(predictions=preds, references=refs)
    return val_loss / len(val_iter), rouge_scores


def tokens_to_text(tokens, field):
    tokens = [token for token in tokens if token not in [PAD_IDX, word_field.vocab.stoi[BOS_TOKEN], word_field.vocab.stoi[EOS_TOKEN]]]
    words = [field.vocab.itos[token] for token in tokens]
    return ' '.join(words)


def visualize_attention(attentions, src_tokens, trg_tokens):
    plt.figure(figsize=(10, 10))
    sns.heatmap(attentions.cpu().numpy(), xticklabels=src_tokens, yticklabels=trg_tokens, cmap='viridis')
    plt.xlabel("Source tokens")
    plt.ylabel("Target tokens")
    plt.show()


def generate_and_show_examples(num_test=5, num_custom=5):
    model.eval()
    print("\n=== Генерация из теста ===")
    with torch.no_grad():
        for i, example in enumerate(val_data):
            if i >= num_test:
                break
            src_tensor = example.source.unsqueeze(0).to(DEVICE)
            generated_tokens, attentions = model.generate(src_tensor, max_len=30, bos_idx=word_field.vocab.stoi[BOS_TOKEN])
            pred_text = tokens_to_text(generated_tokens[0].cpu().numpy(), word_field)
            src_text = tokens_to_text(example.source.cpu().numpy(), word_field)
            trg_text = tokens_to_text(example.target.cpu().numpy(), word_field)

            print(f"Source: {src_text}")
            print(f"Target: {trg_text}")
            print(f"Generated: {pred_text}\n")

    # Собственные примеры (тут нужно подготовить свои примеры в формате токенов)
    print("\n=== Генерация из собственных примеров ===")
    custom_texts = [
        "пример текста для генерации заголовка",
        "еще один тестовый пример",
        "третий пример для проверки",
        "пример четыре",
        "пятый пример"
    ]
    for text in custom_texts[:num_custom]:
        # Токенизировать текст и преобразовать в индексы
        tokens = [word_field.vocab.stoi.get(token, UNK_IDX) for token in text.split()]
        src_tensor = torch.tensor([tokens], device=DEVICE)
        generated_tokens, attentions = model.generate(src_tensor, max_len=30, bos_idx=word_field.vocab.stoi[BOS_TOKEN])
        pred_text = tokens_to_text(generated_tokens[0].cpu().numpy(), word_field)
        print(f"Input: {text}")
        print(f"Generated: {pred_text}\n")


if __name__ == "__main__":
    for epoch in range(N_EPOCHS):
        train_loss = train()
        val_loss, rouge_scores = evaluate_model()

        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"]
        })

        print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"ROUGE Scores: {rouge_scores}")

        current_rouge_l = rouge_scores["rougeL"]
        if current_rouge_l > BEST_ROUGE_L:
            BEST_ROUGE_L = current_rouge_l
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Лучшая модель обновлена ROUGE-L: {BEST_ROUGE_L:.4f}")

    generate_and_show_examples()