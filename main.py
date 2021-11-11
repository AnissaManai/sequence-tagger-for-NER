import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score

def split_text_label(filename):
    f = open(filename)
    split_labeled_text = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                split_labeled_text.append(sentence)
                sentence = []
            continue
        splits = line.split()
        sentence.append([splits[0], splits[-1]])
    if len(sentence) > 0:
        split_labeled_text.append(sentence)
        sentence = []
    return split_labeled_text

split_train = split_text_label("data/train.conll")
split_dev = split_text_label("data/dev.conll")
split_test = split_text_label("data/test.conll")


labelSet = set()
wordSet = set()
# Get words and labels sets
for data in [split_train, split_dev, split_test]:
    for labeled_text in data:
        for word, label in labeled_text:
            labelSet.add(label)
            wordSet.add(word.lower())


# Sort the set to ensure '0' is assigned to 0
sorted_labels = sorted(list(labelSet), key=len)

# Create mapping for labels
label2Idx = {}
for label in sorted_labels:
    label2Idx[label] = len(label2Idx)
idx2Label = {v: k for k, v in label2Idx.items()}
# Create mapping for words
word2Idx = {}
if len(word2Idx) == 0:
    word2Idx["PADDING_TOKEN"] = len(word2Idx)
    word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
for word in wordSet:
    word2Idx[word] = len(word2Idx)

# Convert the labels and sentences to tensors of indices
def createMatrices(data, word2Idx, label2Idx):
    sentences = []
    labels = []
    for split_labeled_text in data:
        wordIndices = []
        labelIndices = []
        for word, label in split_labeled_text:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = word2Idx['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label])
        sentences.append(torch.tensor(wordIndices))
        labels.append(torch.tensor(labelIndices))
    return sentences, labels


train_sentences, train_labels = createMatrices(split_train, word2Idx, label2Idx)
dev_sentences, dev_labels = createMatrices(split_dev, word2Idx, label2Idx)
test_sentences, test_labels = createMatrices(split_test, word2Idx, label2Idx)


def padding(sentences, labels):
    padded_sentences = pad_sequence(sentences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)
    return padded_sentences, padded_labels

# Pad sequences to the same length
train_sentences, train_labels = padding(train_sentences, train_labels)
dev_sentences, dev_labels = padding(dev_sentences, dev_labels)
test_sentences, test_labels = padding(test_sentences, test_labels)


#Load glove embeddings
embeddings_index = {}
EMBEDDING_DIM = 50
words_found = 0
f = open('data/glove.6B.50d.txt', encoding="utf-8")
for line in f:
    values = line.strip().split(' ')
    word = values[0]  # get the word 
    coefs = np.asarray(values[1:], dtype='float32') # get embedding vector
    embeddings_index[word] = coefs
f.close()

#create embedding matrix for the tokens
embedding_matrix = np.zeros((len(word2Idx), EMBEDDING_DIM))
for word, i in word2Idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        words_found += 1


def create_emb_layer(embedding_matrix, non_trainable=False):
    vocab_size, embedding_dim = embedding_matrix.size()
    emb_layer = nn.Embedding(vocab_size, embedding_dim)
    emb_layer.load_state_dict({'weight': embedding_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer


class BiLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, out_dim, embeddings):
        super().__init__()
        self.word_embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size = emb_dim, num_layers = 1, hidden_size = hidden_dim,
                            bidirectional=True, batch_first = True)
        self.hidden_to_tag = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, seq):
        embeddings = self.word_embeddings(seq).view(len(seq), 1, -1)
        lstm_out, _ = self.lstm(embeddings)
        tag_outputs = self.hidden_to_tag(lstm_out.view(len(seq), -1))
        tag_scores = F.log_softmax(tag_outputs, dim = 1)
        return tag_scores


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 100
embeds= create_emb_layer(torch.tensor(embedding_matrix), True)
model = BiLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(label2Idx), embeds)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def train(model, train_data, optimizer, loss_function, data_len):
    train_loss = 0.0
    model.train()

    for sentence, labels in train_data:
        # set the gradients to zero before doing parameter update
        model.zero_grad(set_to_none=True)
        tag_scores = model(sentence)
        loss = loss_function(tag_scores, labels)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
    return train_loss / data_len


def evaluate(model, data, loss_function, data_len):
    loss = 0.0
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        for sentence, labels in data:
            tag_scores = model(sentence)
            loss += loss_function(tag_scores, labels).item()
            _, predicted_tags = torch.max(tag_scores, 1)
            outputs.append(predicted_tags)
            targets.append(labels)

    return loss / data_len, outputs, targets

n_epochs = 20
best_dev_loss = float('inf')

for epoch in range(n_epochs):
    train_loss = train(model, zip(train_sentences, train_labels), optimizer = optimizer, loss_function = loss_function, data_len = len(train_sentences))
    dev_loss, outputs, targets = evaluate(model, zip(dev_sentences, dev_labels), loss_function = loss_function, data_len = len(dev_sentences))
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    f1 = f1_score(y_true = targets, y_pred = outputs, average= 'macro')
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    print(f'Epoch \t\t Training Loss: {train_loss:.3f} \t\t Validation Loss: {dev_loss:.3f} \t\t f1_score: {f1:.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))
test_loss, outputs, targets = evaluate(model, zip(test_sentences, test_labels), loss_function= loss_function, data_len = len(test_sentences))
outputs = np.concatenate(outputs)
targets = np.concatenate(targets)
f1 = f1_score(y_true = targets, y_pred = outputs, average= 'macro')
print(f'Test Loss: {test_loss:.3f} \t\t f1_score: {f1:.3f}')

