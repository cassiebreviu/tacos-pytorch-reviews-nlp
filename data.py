
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import TextClassificationDataset
from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from tqdm import tqdm
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader
import pandas as pd
import torch
import io
import os

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


def get_data(batch_size, num_workers):
    target = './.data/yelp_review_full_csv'
    if not os.path.exists(target):
        print('downloading {} ...'.format(target))
        # check directory for data if it doesnt already exist
        if not os.path.isdir('./.data'):
            os.mkdir('./.data')
        #Get train and text dataset to tensor
        yelp_train_dataset, yelp_test_dataset = text_classification.DATASETS['YelpReviewFull'](
            root='./.data', ngrams=2, vocab=None)

        print(f'labels: {yelp_train_dataset.get_labels()}')

        train_sampler = torch.utils.data.distributed.DistributedSampler(yelp_train_dataset)
        train_loader = DataLoader(
        yelp_train_dataset,
        batch_size = batch_size, shuffle=False,
        num_workers= num_workers, pin_memory=True, sampler=train_sampler)


        test_loader = DataLoader(
            yelp_test_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader
    else:
        print('{} already exists, skipping step'.format(str(target)))
        train_csv_file = "./.data/yelp_review_full_csv/train.csv"
        test_csv_file = "./.data/yelp_review_full_csv/test.csv"
        yelp_train_dataset, yelp_test_dataset  = setup_datasets(train_csv_file, test_csv_file, ngrams=2)
        train_sampler = torch.utils.data.distributed.DistributedSampler(yelp_train_dataset)
        train_loader = DataLoader(
        yelp_train_dataset,
        batch_size = batch_size, shuffle=False,
        num_workers= num_workers, pin_memory=True, sampler=train_sampler)

        test_loader = DataLoader(
            yelp_test_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader

def setup_datasets(train_csv_path, test_csv_path, root='.data', ngrams=1, vocab=None, include_unk=False):

    if vocab is None:
        print('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    print('Vocab has {} entries'.format(len(vocab)))
    print('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    print('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))

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


get_data(batch_size = 16, num_workers = 4)
   