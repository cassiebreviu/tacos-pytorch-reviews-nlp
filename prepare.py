
import pickle
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import TextClassificationDataset
from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import get_tokenizer, ngrams_iterator
from tqdm import tqdm
import argparse
from torchtext.datasets import text_classification
from pathlib import Path
import pandas as pd
import torch
import io
import os
import sys

def get_processed_dataset(input_path, mounted_output_path, ngrams):
   
    yelp_train_dataset = setup_datasets(csv_path=input_path, ngrams=ngrams)
    
    vocab = yelp_train_dataset.get_vocab()
    pickle_path = os.path.join(mounted_output_path, './vocab.pickle')
    save_vocab(vocab, pickle_path)

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
            # add gender to dataset
            df['gender'] = df.apply(addGender, axis=1)
            # save file
            filename = f'{files_created}_prepared_data.parquet'
            path = os.path.join(mounted_output_path, filename)
            df.to_parquet(path)
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
    train_data, train_labels = _create_data_from_iterator(vocab, _csv_iterator(csv_path, ngrams, yield_cls=True), include_unk)

    return (TextClassificationDataset(vocab, train_data, train_labels))


def addGender(df):
    if df['label'] >= 3:
        return 'F'
    else:
        return 'M'

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

###################################################################
# Main                                                            #
###################################################################


def main(input_path, output_path):

    print(f'input path: {input_path}')
    print(f'input path: {output_path}')
    get_processed_dataset(input_path, output_path,ngrams=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-s', '--source_path', help='source directory')
    parser.add_argument('-t', '--target_path', help='target path')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    input_path = args.source_path
    output_path = args.target_path
    print(f'input_path: {input_path}')
    print(f'output_path: {output_path}')

    main(input_path, output_path)
