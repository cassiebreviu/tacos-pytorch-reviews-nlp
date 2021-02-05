
import pickle
from torchtext.datasets import text_classification
import pandas as pd
import os
import sys
from data_helper import setup_datasets

def get_processed_dataset(input_path, output_path, ngrams):
   
    yelp_train_dataset = setup_datasets(csv_path=input_path, ngrams=ngrams)
    
    vocab = yelp_train_dataset.get_vocab()
    save_vocab(vocab, './vocab.pickle')

    print("creating dataframe")
    for data_item in enumerate(yelp_train_dataset):
        df = df.append({
                "label": data_item[1][0],
                "tensor":  data_item[1][1].numpy().T
                }, ignore_index=True)

    #df.to_csv('train_tensor_data.csv', index = True)
    df.to_csv(output_path, index = True)
    print("dataframe saved to csv")
    # #save csv
    # o = open(output_path, "w")
    # o.write(csv_processed_data)
    # o.close()

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

mounted_input_path = sys.argv[1]
mounted_output_path = sys.argv[2]
os.makedirs(mounted_output_path, exist_ok=True)

#mounted_input_path = os.path.join(mounted_input_path, 'train.csv')
#mounted_output_path = os.path.join(mounted_output_path, 'processed.csv')

os.join(mounted_input_path, 'train.csv')
os.join(mounted_output_path, 'train.csv')

get_processed_dataset(mounted_input_path, mounted_output_path, ngrams=2)
