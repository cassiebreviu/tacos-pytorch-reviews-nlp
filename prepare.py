
import pickle
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import TextClassificationDataset
from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from tqdm import tqdm
from torchtext.datasets import text_classification
from pathlib import Path
import pandas as pd
import torch
import io
import os
import sys

def get_processed_dataset(input_path, ngrams):
   
    yelp_train_dataset = setup_datasets(csv_path=input_path, ngrams=ngrams)
    
    vocab = yelp_train_dataset.get_vocab()
    save_vocab(vocab, './vocab.pickle')

    print("creating dataframe")
    df = pd.DataFrame(columns=["label", "tensor"])
    files_created = 0
    for i, data_item in enumerate(yelp_train_dataset):
        df = df.append({
                "label": data_item[0],
                "tensor":  data_item[1].numpy().T
                }, ignore_index=True)
        
        # save a file for every 65000 records
        if i % 65000 == 0 and i != 0:
            # save file
            filename = f'{files_created}_prepared_data.parquet'
            df.to_parquet(filename)
            print(f'{filename} - dataframe saved to parquet')
            # reset dataframe
            df = pd.DataFrame(columns=["label", "tensor"])
            files_created += 1

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


def setup_datasets(csv_path, ngrams=2, vocab=None, include_unk=False):

    if vocab is None:
        print('Building Vocab based on {}'.format(csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    
    print('Vocab has {} entries'.format(len(vocab)))
    print('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(csv_path, ngrams, yield_cls=True), include_unk)
    # print('Creating testing data')
    # test_data, test_labels = _create_data_from_iterator(
    #     vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    # if len(train_labels ^ test_labels) > 0:
    #     raise ValueError("Training and test labels don't match")
    # print(type(Vocab))
    # print(type(train_data))
    # print(type(train_labels))

    return (TextClassificationDataset(vocab, train_data, train_labels))
    # print(type(result))
    # # o = open(output_csv_path, "w")
    # # o.write(result)
    # # o.close()
  
    # return (TextClassificationDataset(vocab, train_data, train_labels),
    #         TextClassificationDataset(vocab, test_data, test_labels))

def addGender(df):
    if df['label'] >= 3:
        return 'F'
    else:
        return 'M'

def get_df():
    #File path to the csv file
    csv_file = "./.data/yelp_review_full_csv/train.csv"

    # Read csv file into dataframe
    df = pd.read_csv(csv_file, names=["label", "review"])
    df['gender'] = df.apply(addGender, axis=1)
    # Print first 5 rows in the dataframe
    print(df.head())
    print(df['label'].value_counts())
    print(df['gender'].value_counts())
    return df


def _csv_iterator(data_path, ngrams, yield_cls=False):
    tokenizer = get_tokenizer("basic_english")

    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            tokens = ' '.join(row[1:])
            tokens = tokenizer(tokens)
            if yield_cls:
                yield int(row[0]) - 1, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)
                
def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                        for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                print('Row contains no tokens.')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)



mounted_input_path = sys.argv[1]
#mounted_input_path = check_dir(".data\yelp_review_full_csv")

print(f'input path: {mounted_input_path}')
get_processed_dataset(mounted_input_path, ngrams=2)
