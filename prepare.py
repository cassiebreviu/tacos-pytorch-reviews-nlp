
import pickle
from torchtext.datasets import text_classification
import pandas as pd
import os
from data_helper import setup_datasets

def get_processed_dataset(input_path, output_path, ngrams):
    # check directory for data if it doesnt already exist
    if not os.path.isdir(mounted_input_path):
        os.mkdir(mounted_input_path)
    

    ##TODO change this to input path data csv
    yelp_train_dataset = setup_datasets(csv_path=input_path)
    #Get train and text dataset to tensor
    #yelp_train_dataset, yelp_test_dataset = text_classification.DATASETS['YelpReviewFull'](
    #    root=mounted_input_path, ngrams=2, vocab=None)
    
    vocab = yelp_train_dataset.get_vocab()
    save_vocab(vocab, './vocab.pickle')

    train_dataset_to_list = list(yelp_train_dataset)
    processed_data_df = {'label': [train_dataset_to_list[0]],
                        'tensor': [train_dataset_to_list[1]]} 
    df = pd.DataFrame(processed_data_df)
    print(df)

    for data_item in enumerate(yelp_train_dataset):
        df = df.append({
                "label": data_item[1][0],
                "tensor":  data_item[1][1].numpy().T
                }, ignore_index=True)

    df.to_csv('train_tensor_data.csv', index = True)

    # #save csv
    # o = open(output_path, "w")
    # o.write(csv_processed_data)
    # o.close()

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)


mounted_input_path = '.\.data\yelp_review_full_csv' #sys.argv[1]
mounted_output_path = '.\.data\yelp_review_full_csv' #sys.argv[2]
os.makedirs(mounted_output_path, exist_ok=True)

input_csv_path = os.path.join(mounted_input_path, 'train.csv')
output_csv_path = os.path.join(mounted_output_path, 'processed.csv')


get_processed_dataset(input_csv_path, output_csv_path, ngrams=2)
