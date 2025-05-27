import json
import dill
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from gensim.models import KeyedVectors
from torchtext.data import Field, Example, Dataset, BucketIterator

from data_init import DEVICE, BOS_TOKEN, EOS_TOKEN, PAD_IDX, UNK_IDX


def save_dataset(dataset, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(Path(path) / "dataset.Field", "wb") as f:
        dill.dump(dataset.fields, f)
    with open(Path(path) / "dataset.Example", "wb") as f:
        dill.dump(dataset.examples, f)


def load_dataset(path):
    with open(path + "/dataset.Field", "rb") as f:
        test_fields = dill.load(f)
    with open(path + "/dataset.Example", "rb") as f:
        test_examples = dill.load(f)
    return Dataset(test_examples, test_fields)


def save_word_field(field, path):
    with open(path + "/word_field.Field", "wb") as f:
        dill.dump(field, f)


def load_word_field(path):
    with open(path + "/word_field.Field", "rb") as f:
        return dill.load(f)


def load_rusvectores(emb_path, word_field):
    print("Загрузка RusVectores эмбеддингов...")
    w2v_model = KeyedVectors.load_word2vec_format(
        emb_path,
        binary=False,
        unicode_errors='ignore'
    )

    vectors = []
    oov_count = 0

    for word in word_field.vocab.itos:
        if word in w2v_model:
            vectors.append(torch.tensor(w2v_model[word]))
        else:
            vectors.append(torch.zeros(300))
            oov_count += 1

    print(f"OOV слов: {oov_count}/{len(word_field.vocab)} ({oov_count / len(word_field.vocab):.1%})")
    word_field.vocab.set_vectors(
        stoi=word_field.vocab.stoi,
        vectors=torch.stack(vectors),
        dim=300
    )
    return word_field


def data_preparing(use_rusvectores=True):
    word_field = Field(
        tokenize='moses',
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        batch_first=True,
        lower=True
    )
    fields = [('source', word_field), ('target', word_field)]

    data = pd.read_csv('../data/news.csv', delimiter=',')

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Обработка данных"):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))

    dataset = Dataset(examples, fields)
    train_dataset, val_dataset = dataset.split(split_ratio=0.9)
    train_dataset, test_dataset = train_dataset.split(split_ratio=0.89)

    print("\nПостроение словаря...")
    word_field.build_vocab(train_dataset, min_freq=7)

    if use_rusvectores:
        rusvectores_path = "../data/embeddings/ruwikiruscorpora_upos_skipgram_300_2_2018.vec"
        word_field = load_rusvectores(rusvectores_path, word_field)

    print(f"\nРазмер словаря: {len(word_field.vocab)}")
    #print(f"Пример для 'привет': {word_field.vocab.vectors[word_field.vocab.stoi['привет']][:5]}")

    save_word_field(word_field, "../data")
    save_dataset(train_dataset, "../data/train")
    save_dataset(val_dataset, "../data/val")
    save_dataset(test_dataset, "../data/test")

    with open("datasets_sizes.json", "w") as f:
        json.dump({
            "Train size": len(train_dataset),
            "Validation size": len(val_dataset),
            "Test size": len(test_dataset),
            "Vocab size": len(word_field.vocab)
        }, f, indent=2)

    return train_dataset, val_dataset, test_dataset, word_field


if __name__ == "__main__":
    print(f"Используется устройство: {DEVICE}")
    print("Подготовка данных...")
    train_dataset, val_dataset, test_dataset, word_field = data_preparing(use_rusvectores=True)
    print("Подготовка данных окончена")