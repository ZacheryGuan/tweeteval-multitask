import nltk

nltk.download('punkt')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from collections import defaultdict, OrderedDict
import time
from sklearn.metrics import f1_score, accuracy_score, recall_score
from utils.constants import task_to_task_id, tid_2_name
from model.multitask import MultiTaskTransformerClassification
from dataset.preprocess import preprocess, get_vocabulary
from dataset.multitask import MultiTaskDataset, MultiTaskBatchSampler, DataLoader, pad_sequence
import argparse

parser = argparse.ArgumentParser(description='Multitask Tweet Classification.')
parser.add_argument('--tasks', nargs='*', type=int, help='integer list for task ids')
args = parser.parse_args()
tasks = args.tasks
if tasks is None or len(tasks) == 0:  # use all tasks
    tasks = (0, 1, 2, 3, 4, 5, 6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_data = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
preprocess(all_data, tasks)
vocabulary = get_vocabulary(all_data)
vocab_size = len(vocabulary) + 1
tkn2idx = {w: i for i, w in enumerate(vocabulary, start=1)}
print("Vocabulary Size: " + str(vocab_size))

train_data_combined = [task_train_data for task_name, task_train_data in all_data["train"].items()]
val_data_combined = [task_val_data for task_name, task_val_data in all_data["val"].items()]
test_data_combined = [task_test_data for task_name, task_test_data in all_data["test"].items()]

PAD_IDX = 0
tkn_text_pipeline = lambda words: [tkn2idx[word] for word in words if word in vocabulary]


def collate_batch(batch):
    '''
    input: List[(label, sentence)]
    '''
    label_list, text_list, seq_len, categories = [], [], [], []

    for category, label, words in batch[0]:
        processed_text = torch.tensor(tkn_text_pipeline(words), dtype=torch.int64)
        if processed_text.shape[0] == 0:  # skip empty sentences
            continue
        categories.append(category)
        label_list.append(label)
        text_list.append(processed_text)
        seq_len.append(processed_text.shape[0])
    if len(text_list) != 0:
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, padding_value=PAD_IDX)
        seq_len = torch.tensor(seq_len, dtype=torch.int64)
        categories = torch.tensor(categories, dtype=torch.int64)
    return categories.to(device), label_list.to(device), text_list.to(device), seq_len.to(device)


BATCH_SIZE = 64

train_dataloader = DataLoader(MultiTaskDataset(train_data_combined),
                              sampler=MultiTaskBatchSampler(datasets=train_data_combined, batch_size=BATCH_SIZE),
                              collate_fn=collate_batch)
val_dataloader = DataLoader(MultiTaskDataset(val_data_combined),
                            sampler=MultiTaskBatchSampler(datasets=val_data_combined, batch_size=BATCH_SIZE),
                            collate_fn=collate_batch)
test_dataloader = DataLoader(MultiTaskDataset(test_data_combined),
                             sampler=MultiTaskBatchSampler(datasets=test_data_combined, batch_size=BATCH_SIZE),
                             collate_fn=collate_batch)
cnt = 0
for idx, (category, label, text, seq_len) in enumerate(train_dataloader):
    # print sample batch
    # if idx == 1:
    #     print(category, label, text, seq_len)
    #     print(text.shape)
    cnt = idx

print("Total number of batches: ", cnt)


def train(dataloader, log_interval=5, tasks=(0, 1, 2, 3, 4, 5, 6)):
    model.train()
    start_time = time.time()
    labels = OrderedDict()
    predictions = OrderedDict()

    for task_id in tasks:
        labels[task_id] = []
        predictions[task_id] = []

    for idx, (category, label, text, seq_len) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, seq_len, category[0])
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        curr_predictions = model(text, seq_len, category[0])
        predictions[int(category[0])].extend(curr_predictions.argmax(1).cpu())
        labels[int(category[0])].extend(label.cpu())

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| loss {:5.3f}'.format(epoch, idx, len(dataloader),
                                          loss.item()))

            accuracy = {}
            for task_id in tasks:
                if len(predictions[task_id]) != 0:
                    accuracy[task_id] = accuracy_score(predictions[task_id], labels[task_id])
                else:
                    accuracy[task_id] = -1  # Undefined, no sample of this task_id is in the batch
            # for task_id in range(7):
            #   labels[task_id] = []
            #   predictions[task_id] = []
            print("Train Accuracy: ", ", ".join([f"{task_id}:{acc:.4f}" for task_id, acc in accuracy.items()]))

            start_time = time.time()


def evaluate_test_and_validation(dataloader):
    labels = defaultdict(list)
    predictions = defaultdict(list)
    with torch.no_grad():
        for idx, (category, label, text, seq_len) in enumerate(dataloader):
            category = category.cpu()
            curr_predictions = model(text, seq_len, category[0])
            predictions[int(category[0])].extend(curr_predictions.argmax(1).cpu())
            labels[int(category[0])].extend(label.cpu())

    scores = OrderedDict()
    accuracy = OrderedDict()
    for category in predictions:
        if category == 3:  # Irony
            scores[category] = f1_score(predictions[category], labels[category], average='binary')
        elif category == 5:  # Sentiment
            scores[category] = recall_score(predictions[category], labels[category], average='macro')
        elif category == 6:  # Stance
            labels[category] = np.array(labels[category])
            predictions[category] = np.array(predictions[category])
            labels_a = labels[category][labels[category] != 2]
            predictions_a = predictions[category][labels[category] != 2]
            labels_f = labels[category][labels[category] != 1]
            predictions_f = predictions[category][labels[category] != 1]
            f1_a = f1_score(labels_a, predictions_a, average='macro')
            f1_f = f1_score(labels_f, predictions_f, average='macro')

            scores[category] = 0.5 * (f1_a + f1_f)
        else:
            scores[category] = f1_score(predictions[category], labels[category], average='macro')

        accuracy[category] = accuracy_score(predictions[category], labels[category])

    return scores, accuracy


# use embedding layer
EPOCHS = 1
LR = 5e-4
EMBED_DIM = 300
N_HEADS = 6
DENSE_DIM = 128
N_LAYERS = 1
OPT_STEP = 10
OPT_GAMMA = 0.75

model = MultiTaskTransformerClassification(maxlen=5000, vocab_size=vocab_size, embed_dim=EMBED_DIM,
                                           n_heads=N_HEADS, attn_drop_rate=0.1, layer_drop_rate=0.1,
                                           dense_dim=DENSE_DIM, n_layers=N_LAYERS).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, OPT_STEP, gamma=OPT_GAMMA)

print(f"Current hparams: "
      f"EPOCHS={EPOCHS}, LR={LR}, EMBED_DIM={EMBED_DIM}, "
      f"N_HEADS={N_HEADS}, DENSE_DIM={DENSE_DIM}, N_LAYERS={N_LAYERS}, "
      f"BATCH_SIZE={BATCH_SIZE}, OPT_STEP={OPT_STEP}, OPT_GAMMA={OPT_GAMMA}.")

# training procedure
cur_score, best_model = 0, None
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader, log_interval=10, tasks=tasks)
    validation_score, validation_accuracy = evaluate_test_and_validation(val_dataloader)
    average_score = sum(validation_score.values()) / len(validation_score)
    scheduler.step()
    if average_score > cur_score:  # save best model
        cur_score = average_score
        best_model = model.state_dict()
        torch.save(model, "best.pt")

    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s'.format(epoch,
                                                         time.time() - epoch_start_time))

    print("Category:\t", "\t".join([str(category) for category in sorted(validation_score)]))
    print("Accuracy:", " ".join([f"{validation_accuracy[category]:7.4f}" for category in sorted(validation_score)]))
    print("Score:   ", " ".join([f"{validation_score[category]:7.4f}" for category in sorted(validation_score)]))
    print("Average score: %.4f" % average_score)
    print("Current LR: %e" % optimizer.param_groups[0]['lr'])

    print('-' * 59)

# test procedure
model.load_state_dict(best_model)
test_score, test_accuracy = evaluate_test_and_validation(test_dataloader)

print("Category:\t", " ".join([f"{tid_2_name[category]:7}" for category in sorted(test_score)]))
print("Accuracy:", " ".join([f"{test_accuracy[category]:7.4f}" for category in sorted(test_score)]))
print("Score:   ", " ".join([f"{test_score[category]:7.4f}" for category in sorted(test_score)]))
print("Average score:", sum(test_score.values()) / len(test_score))
