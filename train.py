import os
import time
import argparse
from pathlib import Path
from azureml.core.run import Run
import torch
from torchtext.data.utils import ngrams_iterator, get_tokenizer, ngrams_iterator
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import TextClassificationDataset
from torch.utils.data.dataset import random_split
import pickle
import pandas as pd
import torch.onnx as onnx
import mlflow
from azureml.core import Workspace, Experiment, Model 
import joblib

###################################################################
# Helpers                                                         #
###################################################################

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    #gender = torch.tensor([entry[2] for entry in batch])
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def split_gender_features(data):
    # label = [entry[0] for entry in data]
    # text = [entry[1] for entry in data]
    gender = [entry[2] for entry in data]
    #return [label, text], gender
    return gender


def load_vocab(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
        return vocab

def array_to_tensor(df_item):
    return torch.tensor(df_item['tensor'])

def gender_to_int(df_item):
    if df_item['gender'] == 'F':
        return 0
    else:
        return 1

def save_model_onnx(model, x_input_shape, model_output_path):
    model.eval()
    # Export the model
    onnx.export(model,         # model being run
                x_input_shape,       # model input (or a tuple for multiple inputs)
                model_output_path,         # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=11,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                              'output' : {0 : 'batch_size'}})

  
def load_model_onnx(model_path):
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model

def register_model(name, model_path, run):
    print("Registering ", name)
    if run != None:
        run.upload_file(name, str(model_path))
            
        model = run.register_model(model_name='tacoreviewsmodel', 
                            model_path=name, 
                            model_framework="PyTorch", 
                            model_framework_version=torch.__version__, 
                            description="Review sentiment model")

        print("Registered ", model.id)
        return model.id
    else:
        print('No active run avaiable to register')
        return None

def fairness_dashboard(sensitive_feature_df, yelp_test_dataset):

    from fairlearn.widget import FairlearnDashboard
    #TODO split out test dataset to features and labels
    X_test, y_test = yelp_test_dataset, yelp_test_dataset
    FairlearnDashboard(sensitive_features=sensitive_feature_df, 
                    sensitive_feature_names=['gender'],
                    y_true=y_test,
                    y_pred={"lr_model": predict(text, model, vocab, ngrams=2)})


###################################################################
# Training                                                        #
###################################################################

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return F.softmax(self.fc(embedded), dim=1)

def train_func(train_dataset, batch_size, optimizer, model, criterion, scheduler, device):

    # Train the model
    train_loss = 0
    train_acc = 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

    for _, (text, offsets, label) in enumerate(train_loader):
        optimizer.zero_grad()
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        output = model(text, offsets)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()

        # logging batch metrics
        batch_loss = loss.item()
        batch_acc = (output.argmax(1) == label).sum().item()

        #mlflow.log_metric("batch_loss", batch_loss)
        #mlflow.log_metric("batch_acc", batch_acc / float(len(cls)))

        # accumulating epoch metrics
        train_loss += batch_loss
        train_acc += batch_acc

    # Adjust the learning rate
    scheduler.step()
    items_in_dataset = len(train_dataset)

    train_loss = train_loss / items_in_dataset
    train_acc = train_acc / items_in_dataset
    print(f'train_loss: {train_loss}, train_acc: {train_acc}')
    return train_loss, train_acc

###################################################################
# Testing                                                         #
###################################################################

def test(test_dataset, batch_size, model, criterion, device):
    loss = 0
    acc = 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, collate_fn=generate_batch)

    for text, offsets, label in test_loader:
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()

    items_in_dataset = len(test_dataset)

    val_loss = float(loss) / items_in_dataset
    val_acc = acc / items_in_dataset
    return val_loss, val_acc


def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

###################################################################
# Main                                                            #
###################################################################

def main(input_path, output_path, device, run, epochs, lr, batch_size):
    info('Data')
    # Get data
    # dataset object from the run
    #run = Run.get_context()
    print(f'input_path: {input_path}')
    print(f'input_path: {output_path}')
    #get all files from directory
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    print(f'input_path: {input_path}')
    print(f'input_path: {output_path}')

    # Create df and vocab
    train_df = pd.DataFrame(columns=['label', 'tensor'])
    vocab = None

    # to shorten training time
    parquet_files = 10000

    # loop thru files and create df and vocab
    for file in os.listdir(str(input_path)):
        print(f'\t{file}')
        if file.__contains__('parquet'):
            if parquet_files > 0:
                print(file)
                df = pd.read_parquet(Path(os.path.join(input_path, file)).resolve())
                print(len(df))
                train_df = train_df.append(df)
                print(f'length of loaded train_df: {len(train_df)}')
                parquet_files -= 1
        else:
            vocab = load_vocab(Path(os.path.join(input_path,'vocab.pickle')).resolve())

  
    #create tensor and remove header row
    #train_df['gender'] = train_df.apply(gender_to_int, axis=1)
    train_df['tensor'] = train_df.apply(array_to_tensor, axis=1)
    train_data = list(train_df.values)
    train_labels = set(train_df['label'])
    print(train_df.head())
    print(train_labels)

    full_dataset = TextClassificationDataset(vocab, train_data, train_labels)
    

    #gets 80% of length of data to split 80/20 for training
    train_len = int(len(full_dataset) * 0.80)
    
    yelp_train_dataset, yelp_test_dataset = random_split(full_dataset, [train_len, len(full_dataset) - train_len])
    # slit off gender from train dataset here?

    # #split out text and label from gender
    # train_sensitive_feature_gender = split_gender_features(yelp_train_dataset)
    # test_sensitive_feature_gender = split_gender_features(yelp_test_dataset)
    # fairness_dashboard(test_sensitive_feature_gender, yelp_test_dataset)

    # print(f'gender train data {len(train_sensitive_feature_gender)}')
    # print(f'gender test data {len(test_sensitive_feature_gender)}')

    VOCAB_SIZE = len(full_dataset.get_vocab())
    EMBED_DIM = 32
    batch_size = batch_size
    NUM_CLASS = len(train_labels)

    print(f'create model VOCAB_SIZE: {VOCAB_SIZE} NUM_CLASS: {NUM_CLASS}')
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

    N_EPOCHS = epochs

    #min_valid_loss = float('inf')

    # loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # using Adam
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(yelp_train_dataset) * 0.95)
    train_split_data, valid_split_data = random_split(yelp_train_dataset, [train_len, len(yelp_train_dataset) - train_len])

    info('Training')

    mlflow.start_run()
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train_func(train_split_data, batch_size, optimizer, model, criterion, scheduler, device)
        valid_loss, valid_acc = test(valid_split_data, batch_size, model, criterion, device)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

        # log to mlflow
        mlflow.log_metric("epoch_loss", train_loss)
        mlflow.log_metric("epoch_accuracy", train_acc)
        mlflow.log_metric("val_loss", valid_loss)
        mlflow.log_metric("val_acc", valid_acc)

    
    info('Test')

    test_loss, test_acc = test(yelp_test_dataset,batch_size, model, criterion, device)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_acc", test_acc)

    mlflow.end_run()

    # Saving model files
    print(f'Output path => {str(output_path)}')

    model_name = "model.pth"
    vocab_name = "vocab.data"

    file_output = (output_path / model_name).resolve()
    print(f'Writing model to {str(file_output)}... ', end='')
    torch.save(model.to('cpu'), str(file_output))
    print('Done!')

    vocab_file = (output_path / vocab_name).resolve()
    print(f'Writing vocab to {str(vocab_file)}... ', end='')
    torch.save(full_dataset.get_vocab(), str(vocab_file))
    print("Done!")

    if run != None:
        print("Registering model")
        folder = 'model_files'
        run.upload_folder(name=folder, path=str(output_path))
        model = run.register_model(model_name='tacoreviewsmodel', 
                            model_path=folder, 
                            model_framework="PyTorch", 
                            model_framework_version=torch.__version__, 
                            description="Review sentiment model")
        print(f'Registered {model.id}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-s', '--source_path', help='source directory')
    parser.add_argument('-t', '--target_path', help='target path')
    parser.add_argument('-e', '--epochs', help="number of epochs", type=int, default=3)
    parser.add_argument('-l', '--learning_rate', help="learning rate", type=float, default=4.0)
    parser.add_argument('-b', '--batch_size', help="batch_size", type=int, default=16)
    parser.add_argument('-n', '--experiment_name', help='experiment name', default='nlp-sentiment-reviews-train')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    input_path = args.source_path
    output_path = args.target_path
    print(f'input_path: {input_path}')
    print(f'output_path: {output_path}')

    lr = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    print(f'lr: {lr} epochs: {epochs} batch_size: {batch_size}')


    # Get run info.
    run = Run.get_context()
    print('AML Context: {}'.format(run.id))
    offline = run.id.startswith('OfflineRun')
    if offline:
        run = None

    #setup mlflow
    if run != None:
        mlflow.set_tracking_uri(run.experiment.workspace.get_mlflow_tracking_uri())

    mlflow.set_experiment(args.experiment_name)

    main(input_path, output_path, device, run, epochs, lr, batch_size)

# Resources:
# This example is from the [PyTorch Beginner Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

