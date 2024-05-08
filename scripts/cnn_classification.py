import collections

import datasets

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import argparse
from sklearn.metrics import f1_score
seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


data_files = {}
data_files["train"] = "scripts/data/medabstracts/train.csv"
data_files["validation"] = "scripts/data/medabstracts/test.csv"

class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, n_filters, filter_size)
                for filter_size in filter_sizes
            ]
        )
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, embedding dim, seq len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [conv.max(dim=-1).values for conv in conved]
        # pooled_n = [batch size, n filters]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        # cat = [batch size, n filters * len(filter_sizes)]
        prediction = self.fc(cat)
        # prediction = [batch size, output dim]
        return prediction


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens}

def map_labels(example,label_to_id):
    labels = label_to_id[example['label']]
    return {"labels": labels}

def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["labels"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "labels": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader  

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    epoch_f1 = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["labels"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        f1 = f1_score(label, prediction, average='micro')
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
        epoch_f1.append(f1)
    return np.mean(epoch_losses), np.mean(epoch_accs), np.mean(epoch_f1)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    epoch_f1 = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["labels"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            f1 = f1_score(label, prediction, average='micro')
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
            epoch_f1.append(f1)

    return np.mean(epoch_losses), np.mean(epoch_accs), np.mean(epoch_f1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--validation_file",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="Learning rate used for training.",
    )
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="number of trianing epochs."
    )

    args = parser.parse_args()

    data_files = {}
    data_files["train"] = args.train_file
    data_files["validation"] = args.validation_file

    data = datasets.load_dataset("csv", data_files=data_files)

    train_data = data['train']
    test_data = data['validation']

    label_list = train_data.unique("label")
    label_list.sort()

    label_to_id = None
 
    label_to_id = {v: i for i, v in enumerate(label_list)}

    id_to_label = {id: label for label, id in label_to_id.items()}

    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

    max_length = args.max_seq_length

    train_data = train_data.map(
        tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )
    test_data = test_data.map(
        tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
    )

    train_data = train_data.map(
    map_labels, fn_kwargs={"label_to_id": label_to_id}
    )

    test_data = test_data.map(
        map_labels, fn_kwargs={"label_to_id": label_to_id}
    )

    test_size = 0.25

    train_valid_data = train_data.train_test_split(test_size=test_size)
    train_data = train_valid_data["train"]
    valid_data = train_valid_data["test"]

    min_freq = 5
    special_tokens = ["<unk>", "<pad>"]

    vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["tokens"],
        min_freq=min_freq,
        specials=special_tokens,
    )

    unk_index = vocab["<unk>"]
    pad_index = vocab["<pad>"]

    vocab.set_default_index(unk_index)

    train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
    test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})

    train_data = train_data.with_format(type="torch", columns=["ids", "labels"])
    valid_data = valid_data.with_format(type="torch", columns=["ids", "labels"])
    test_data = test_data.with_format(type="torch", columns=["ids", "labels"])

    train_batch_size = args.per_gpu_train_batch_size

    train_data_loader = get_data_loader(train_data, args.per_gpu_train_batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, args.per_gpu_eval_batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, args.per_gpu_eval_batch_size, pad_index)


    vocab_size = len(vocab)
    embedding_dim = 300
    n_filters = 100
    filter_sizes = [3, 4, 5, 7]
    output_dim = len(train_data.unique("labels"))
    dropout_rate = 0.25

    model = CNN(
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout_rate,
        pad_index,
    )

    print(f"The model has {count_parameters(model):,} trainable parameters")

    model.apply(initialize_weights)
    vectors = torchtext.vocab.FastText()
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 10
    best_valid_loss = float("inf")


    metrics = collections.defaultdict(list)

    for epoch in range(n_epochs):
        train_loss, train_acc, train_f1 = train(
            train_data_loader, model, criterion, optimizer, device
        )
        valid_loss, valid_acc, val_f1 = evaluate(valid_data_loader, model, criterion, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["train_f1"].append(train_f1)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        metrics["val_f1"].append(val_f1)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "cnn.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, train_f1: {train_f1:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}, valid_f1: {val_f1:.3f}")


if __name__ == "__main__":

    main()