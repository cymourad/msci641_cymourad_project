# The goal of this script is to use bert to get a sentence embedding for each overview.

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import os
from torch.utils.data import DataLoader
import csv
from tqdm import tqdm

from data_prep import get_distil_data
from features import get_top_n_per_feature, parse_into_python_objects


# CONSTANTS

DEV_MODE = True
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
N_EPOCHS = 5
PATH_TO_MODEL_PICKELING_DIRECTORY = './bert_model'
SENTENCE_EMBEDDING_FILE = 'sentence_embeddings.csv'

# load the data
df = get_distil_data('data/distil_data.csv')

# only keep a small portion of the data if in dev mode
if DEV_MODE:
    print('DEV mode: Only keeping 100 lines and training for 1 epoch ...')
    df = df.head(100)
    N_EPOCHS = 1

# the csv files have stringified objects to represnt the cast, the crew, the genres and the prodiction companies
# we have to parse them into python objects
df = parse_into_python_objects(df, ['genres'])

# let's extract the top 3 genres of a movie into lists (instead of objects)
# df = get_top_n_per_feature(df, [('genres', 3)])

print('One-hot-encoding the genres ...')
# one-hot-encoding for the genres
mlb_genres = MultiLabelBinarizer()

df = df.join(
    pd.DataFrame(
        mlb_genres.fit_transform(df.pop('genres')),
        columns=mlb_genres.classes_,
        index=df.index
    )
)

genre_names = mlb_genres.classes_.tolist()

# create label indices (for classification)
labels = genre_names
id2label = {idx: label for idx, label in enumerate(labels)}
label2id = {label: idx for idx, label in enumerate(labels)}

print('Splitting into train/test sets ...')

# create a mask to split data into training and testing
msk = np.random.rand(len(df)) < 0.8

# drop all the columns that I do not need (i.e., cast, crew, maybe title)
# only keep:
# - id (to identify the movie later)
# - overview (that we will tokenize)
# - labels (that we will use for the classification)
desired_df = df[['id', 'overview'] + labels]

# make a huggingface dataset dictionary to have the train and test sets
dataset = DatasetDict(
    train=Dataset.from_pandas(desired_df[msk]),
    test=Dataset.from_pandas(desired_df[~msk])
)

# find the sentence length for the tokenizer
count = df['overview'].str.split().str.len()
tokens_max_length = count.quantile(0.95)

# make the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def preprocess_data(examples):
    # take a batch of texts
    text = examples["overview"]
    # encode them
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=int(tokens_max_length),
    )
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    encoding["id"] = examples["id"]

    return encoding


print('Pre-processing the overviews ...')

encoded_dataset = dataset.map(
    preprocess_data,
    batched=True,  # default batch size is 1,000
    # the returned values will have a new shape,
    # so we must drop the old columns lest we have shape mismatch problems
    remove_columns=dataset['train'].column_names
)

# set the format of our data to PyTorch tensors.
# This will turn the training, validation and test sets into standard PyTorch datasets
encoded_dataset.set_format("torch")

# create a directory to save the model and train the model (if one is not already created)
if not os.path.exists(PATH_TO_MODEL_PICKELING_DIRECTORY):
    print(f'Creating {PATH_TO_MODEL_PICKELING_DIRECTORY} ...')
    os.makedirs(PATH_TO_MODEL_PICKELING_DIRECTORY)

    print('Creating the model ...')

    # initialize the model from a pre-trained BERT
    # we should tell the model that we will want to extract the embeddings
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        output_hidden_states=True)

    # send the model to the GPU "hopefully"
    # model.to(device)

    # training parameters
    batch_size = 32
    metric_name = "f1"

    args = TrainingArguments(
        f"bert-finetuned-sem_eval-english",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        # push_to_hub=True,
    )

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/

    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(
            y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    print('Creating a trainer ...')

    # train the model
    # by default, the trainer will check if a cuda is available and if so, it will do all
    # the computations there
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print('Starting training ...')

    trainer.train()

    print(f'Saving the model to {PATH_TO_MODEL_PICKELING_DIRECTORY}')
    # save the model weights to that directory to be easily loaded later
    model.save_pretrained(PATH_TO_MODEL_PICKELING_DIRECTORY)

else:
    # if the model is already trained and the weights are saved, we can simply load them
    model = AutoModelForSequenceClassification.from_pretrained(
        PATH_TO_MODEL_PICKELING_DIRECTORY)

# we will now use the model to get an embedding to represent each sentence

print('Starting evaluation mode to extract the embeddings ...')

# since we will be doing this without the help of hugging face classes or functions,
# we need to send the model expicitly to the GPU (it's probably already there)
# TODO the model might already be on the GPU (put there by huggingface earlier)
model.to(device)

# set model to evaluation mode so it does not upadte weights
model.eval()

EXTRACTION_BATCH_SIZE = 10

# Run the text through BERT, and collect all of the hidden states produced
# from the last layer.
with torch.no_grad():
    # batch = encoded_dataset['train'][:10]
    for data_split in ['train', 'test']:

        for batch in tqdm(DataLoader(
                encoded_dataset[data_split], EXTRACTION_BATCH_SIZE)):

            # this evaluates the model (a.k.a. finds the outputs) for the first 10 sentences
            # TODO we will do the same for batches (maybe size 64)
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                labels=batch['labels'].to(device))

            # get the features in the last layer
            # TODO is it better to get the features in the second last layer [-2] than the last layer [-1]
            last_layer = outputs.hidden_states[-2]

            # to get the representation of a sentence, combine all the words embeddings (by averaging them) to become a single sentence embedding
            # in other words, we want to get rid of the middle dimension (the <SENTENCE_LENGTH> dimension), which is at index 1 of the shape
            # the shape is (batch_size, sentence_length, num_features_in_hidden_state)
            sentence_embedding = last_layer.mean(1)

            with open(SENTENCE_EMBEDDING_FILE, 'a', newline='') as f:
                writer = csv.writer(f)

                for sentence, id in zip(sentence_embedding, batch['id']):
                    writer.writerow([id.tolist()] + sentence.tolist())
