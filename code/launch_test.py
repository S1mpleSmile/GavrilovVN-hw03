import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.data import BucketIterator
import evaluate
from tqdm import tqdm


from prelaunch import load_dataset, load_word_field, BOS_TOKEN, EOS_TOKEN
from model import EncoderDecoder
from data_init import DEVICE, tokens_to_words, words_to_tokens, PAD_IDX

import torch.nn as nn


HIDDEN_SIZE = 512
EMB_DIM = 300

MODEL_PATH = "best_model.pt"
MAX_LEN = 50


def build_model(embedding, hidden_size, vocab_size):
    return EncoderDecoder(embedding, hidden_size=hidden_size, output_size=vocab_size).to(DEVICE)


def load_model(word_field):
    vocab_size = len(word_field.vocab)
    embedding = nn.Embedding.from_pretrained(word_field.vocab.vectors, freeze=False, padding_idx=PAD_IDX)
    model = build_model(embedding, HIDDEN_SIZE, vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def tokens_to_text(tokens, field):
    tokens = [t for t in tokens if t not in [PAD_IDX, field.vocab.stoi[BOS_TOKEN], field.vocab.stoi[EOS_TOKEN]]]
    return ' '.join([field.vocab.itos[t] for t in tokens])


def prt_test(model, iterator, word_field):

    predictions, references = [], []
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Вывод"):
            src = batch.source.to(DEVICE)
            trg = batch.target.to(DEVICE)

            generated, _ = model.generate(src, max_len=MAX_LEN, bos_idx=word_field.vocab.stoi[BOS_TOKEN])
            for i in range(src.shape[0]):
                pred = tokens_to_text(generated[i].cpu().numpy(), word_field)
                true = tokens_to_text(trg[i].cpu().numpy(), word_field)
                predictions.append(pred)
                references.append(true)

    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=predictions, references=references)

    print(json.dumps(scores, indent=4, ensure_ascii=False))

    with open("ROUGE_metrics.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)


def graph_attention(attention, src_tokens, trg_tokens, idx):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention.cpu().numpy(), xticklabels=src_tokens, yticklabels=trg_tokens, cmap='viridis')
    plt.title(f"Attention Example {idx}")
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(f"attention_example_{idx}.png")
    plt.close()


def sum_exmp(model, word_field, num_test=5, num_custom=5):
    print("=== Тестовые примеры ===")
    test_data = load_dataset("../data/test")
    for idx, example in enumerate(test_data[:num_test]):
        tokens = [word_field.vocab.stoi.get(tok, word_field.vocab.stoi["<unk>"]) for tok in example.source]
        src_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        generated, attention = model.generate(src_tensor, max_len=MAX_LEN, bos_idx=word_field.vocab.stoi[BOS_TOKEN])

        pred = tokens_to_text(generated[0].cpu().numpy(), word_field)
        src = ' '.join(example.source)  # уже текст
        trg = ' '.join(example.target)  # тоже строки
        print(f"Source: {src}")
        print(f"Target: {trg}")
        print(f"Generated: {pred}\n")
        graph_attention(attention[0], src.split(), pred.split(), idx)

    print("\n=== Пользовательские примеры ===")
    custom_inputs = [
        "пример текста для генерации заголовка",
        "еще один тестовый пример",
        "третий пример для проверки",
        "пример четыре",
        "пятый пример"
    ]
    for i, text in enumerate(custom_inputs[:num_custom]):
        tokens = [word_field.vocab.stoi.get(tok, word_field.vocab.stoi["<unk>"]) for tok in text.split()]
        src_tensor = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
        generated, attention = model.generate(src_tensor, max_len=MAX_LEN, bos_idx=word_field.vocab.stoi[BOS_TOKEN])
        pred = tokens_to_text(generated[0].cpu().numpy(), word_field)
        print(f"Input: {text}")
        print(f"Generated: {pred}\n")
        graph_attention(attention[0], text.split(), pred.split(), f"custom_{i}")


if __name__ == "__main__":
    word_field = load_word_field("../data")
    test_data = load_dataset("../data/test")
    test_iter = BucketIterator(test_data, batch_size=1, device=DEVICE, sort=False, sort_within_batch=False)

    model = load_model(word_field)

    print("=== Оценка модели на тесте ===")
    prt_test(model, test_iter, word_field)

    print("\n=== Демонстрация генерации ===")
    sum_exmp(model, word_field)
