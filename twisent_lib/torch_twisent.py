import random

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data, datasets

from twisent_lib import print_stamp, twisent_tokenizer


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x


class TwisentDataset(data.Dataset):
    def __init__(self, df, fields, **kwargs):
        examples = []

        for index, row in df.iterrows():
            text = row['text']
            target = row['target']
            examples.append(data.Example.fromlist([text, target], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(pred, y):
    rounded_pred = torch.round(torch.sigmoid(pred))
    correct = (rounded_pred == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.target)
        acc = binary_accuracy(predictions, batch.target)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.target)
            acc = binary_accuracy(predictions, batch.target)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        return self.fc(pooled)


if __name__ == "__main__":
    main_start = print_stamp("We begin!")
    print("=================================================")
    print("")

    SEED = 1234
    MAX_VOCAB_SIZE = 25_000
    # VOCAB_VECTOR_SET = "glove.6B.100d"
    VOCAB_VECTOR_SET = "glove.twitter.27B.100d"
    BATCH_SIZE = 64

    # INPUT_DIM will get a value equal to len(TEXT.vocab) later on
    INPUT_DIM = 1
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    # single output dimension for binary scalar value (0, 1)
    OUTPUT_DIM = 1
    # PAD_IDX will get a value after the vocab is loaded
    PAD_IDX = 1

    N_EPOCHS = 5

    MODEL_FILE = "pickle/torch_twisent.pt"

    t = print_stamp("Reading full dataframe...")
    full_df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", header=None, names=["target", "status_id", "datetime", "query", "handle", "text"], encoding="latin-1")
    print_stamp("Reading complete.", t)
    print("=================================================")
    print("")

    RETRAIN_WHOLE = False
    TRUNCATE_ROWS = 0

    t = print_stamp("Modifying dataframe...")

    if TRUNCATE_ROWS > 0:
        print("Truncating rows to {0:,} each of pos and neg".format(TRUNCATE_ROWS))
        full_df = full_df.head(TRUNCATE_ROWS).append(full_df.iloc[800000:800000+TRUNCATE_ROWS, :])

    # convert target=4 to target=1
    print("Replacing 4 with 1 in target col")
    full_df['target'].replace(4, 1, inplace=True)
    # sort chrono
    full_df.sort_values("datetime", ascending=True, inplace=True)

    # we only want to keep fields: target, text
    print("Dropping cols besides target and text")
    full_df.drop(["status_id", "datetime", "query", "handle"], axis=1, inplace=True)

    print_stamp("Dataframe modification complete.", t)
    print("=================================================")
    print("")

    t = print_stamp("Splitting into train and test...")
    train_size = int(full_df.shape[0] * 0.8)
    test_size = full_df.shape[0] - train_size

    train_df = full_df.head(train_size)
    test_df = full_df.tail(test_size)

    print("train_df", train_df.shape)
    print("test_df ", test_df.shape)

    print_stamp("Split complete.", t)
    print("=================================================")
    print("")

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(tokenize='spacy', tokenizer_language="en_core_web_sm", preprocessing=generate_bigrams)
    #TEXT = data.Field(tokenize=twisent_tokenizer, tokenizer_language="en_core_web_sm", preprocessing=generate_bigrams)
    LABEL = data.LabelField(dtype=torch.float)

    df_fieldsX = {
        "text": TEXT,
        "target": LABEL
    }

    df_fields = [('text', TEXT), ('target', LABEL)]

    t = print_stamp("Converting pandas to pytorch...")
    # train_data = DataFrameDataset(train_df, df_fields)
    # test_data = DataFrameDataset(test_df, df_fields)
    train_data = TwisentDataset(train_df, df_fields)
    test_data = TwisentDataset(test_df, df_fields)
    print_stamp("Conversion complete.", t)

    print("=================================================")
    print("")

    #t = print_stamp("Downloading and splitting IMDB dataset...")
    #train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    #print_stamp("Download and split complete.", t)

    print("Exploring the pytorch train data:")
    print("Type of train_data:", type(train_data))
    print("Length of train_data:", len(train_data))
    print("")
    print("Example record from training data:")
    print(vars(train_data.examples[0]))
    #print(train_data.examples[0])

    print("=================================================")
    print("")

    t = print_stamp("Splitting into train/valid...")
    # Split train into train/valid
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    print_stamp("Split complete.", t)
    print("=================================================")
    print("")

    t = print_stamp("Building vocabulary, max=[{0:d}] vectors=[{1:s}]".format(MAX_VOCAB_SIZE, VOCAB_VECTOR_SET))
    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors=VOCAB_VECTOR_SET,
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    print_stamp("Vocabulary complete.", t)
    print("=================================================")
    print("")

    print("Vocabulary stats:")
    print("Unique tokens in TEXT vocab  :", len(TEXT.vocab))
    print("Unique tokens in LABEL vocab :", len(LABEL.vocab))
    print("")
    print("The 20 most common words in the vocab:")
    print(TEXT.vocab.freqs.most_common(20))
    print("")
    print("Top 10 itos:")
    print(TEXT.vocab.itos[:10])
    print("")
    print("Labels stoi:")
    print(LABEL.vocab.stoi)
    print("=================================================")
    print("")

    print("Setting up device, forcing it to 'cpu' because GeForce GT730 is unsupported...")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("Device:", str(device))
    print("=================================================")
    print("")

    t = print_stamp("Setting up bucket iterator...")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device
    )
    print_stamp("Setup complete.", t)
    print("=================================================")
    print("")

    t = print_stamp("Setting up the model...")
    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
    print_stamp("Model setup complete.", t)
    print("The model has {0:,} trainable parameters.".format(count_parameters(model)))
    print("=================================================")
    print("")

    t = print_stamp("Copying pretrained word vectors to model...")
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    print_stamp("Copy complete.", t)
    print("=================================================")
    print("")

    t = print_stamp("Initializing weights of unknown and padding tokens to ZERO...")
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    print_stamp("Zeroing complete.", t)
    print("=================================================")
    print("")

    # Adam uses adaptive learning rate for each parameter, in place of SGD, no initial LR needed
    optimizer = torch.optim.Adam(model.parameters())
    # loss function = binary cross entropy with logits
    criterion = nn.BCEWithLogitsLoss()

    t = print_stamp("Moving model to the device: {0:s}".format(str(device)))
    model = model.to(device)
    criterion = criterion.to(device)
    print_stamp("Move complete.", t)
    print("=================================================")
    print("")

    train_start = print_stamp("Training begins...")

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        t = print_stamp("Begin epoch {0:d}...".format(epoch + 1))

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        print_stamp("Epoch {0:d} complete.".format(epoch + 1), t)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_FILE)
            print("New best validation loss, saving model ({0:s})".format(MODEL_FILE))

        print("")

    print_stamp("Training complete.", train_start)
    print("=================================================")
    print("")

    t = print_stamp("Loading best saved model...")
    model.load_state_dict(torch.load(MODEL_FILE))
    print_stamp("Loading complete.", t)
    print("")
    t = print_stamp("Predicting on test set...")
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print_stamp("Prediction complete.", t)
    print("")

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

    print("")
    print_stamp("Good bye.", main_start)
