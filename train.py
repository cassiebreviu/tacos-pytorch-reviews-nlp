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
import torch.utils.data.distributed
from data import get_data
import pickle
import pandas as pd

###################################################################
# Helpers                                                         #
###################################################################
def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

def load_vocab(filename):
    with open(filename, 'rb') as f:
        vocab = pickle.load(f)
        return vocab

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
        return self.fc(embedded)

def train_func(train_dataset, batch_size,optimizer, model, criterion, scheduler, device, distributed, num_workers):

    # Train the model
    train_loss = 0
    train_acc = 0

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size= batch_size, shuffle=(train_sampler is None),
        num_workers= num_workers, pin_memory=True, sampler=train_sampler)

    for i, (text, offsets, cls) in enumerate(train_loader):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(train_loader), train_acc / len(train_loader)

###################################################################
# Testing                                                         #
###################################################################

def test(test_dataset, batch_size, model, criterion, device, distributed, num_workers):
    loss = 0
    acc = 0


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    for text, offsets, cls in test_loader:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(test_loader), acc / len(test_loader)


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


def main(run, data_path, output_path, log_path, batch_size, epochs, learning_rate, distributed, device, num_workers ):
    info('Data')
    # Get data
    # dataset object from the run
    run = Run.get_context()
    datasets = run.input_datasets['prepared_reviews_ds']

    train_df = pd.DataFrame()
    for dataset in enumerate(datasets):
        df = pd.read_parquet(dataset)
        train_df.append(df)
        

    vocab = load_vocab('vocab.pickle')
    train_data = train_df['tensors']
    train_labels = train_df['labels']

    #yelp_train_dataset, yelp_test_dataset = get_data()

    full_dataset = TextClassificationDataset(vocab, train_data, train_labels)
    (yelp_train_dataset, yelp_test_dataset) = full_dataset.random_split(percentage=0.8, seed=111)

    VOCAB_SIZE = len(yelp_train_dataset.get_vocab())
    EMBED_DIM = 32
    #batch_size = 16
    NUN_CLASS = len(yelp_train_dataset.get_labels())
    model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

    if not distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    N_EPOCHS = epochs
    #min_valid_loss = float('inf')

    #activation function
    criterion = torch.nn.CrossEntropyLoss().to(device)
    #Stochastic Gradient descient with optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_len = int(len(yelp_train_dataset) * 0.95)
    train_split_data, valid_split_data = random_split(yelp_train_dataset, [train_len, len(yelp_train_dataset) - train_len])

    info('Training')


    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train_func(train_split_data, batch_size, optimizer, model, criterion, scheduler, device, distributed, num_workers)
        valid_loss, valid_acc = test(valid_split_data, batch_size, model, criterion, device, num_workers)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    #file_output = os.path.join(output_path, 'latest.hdf5')
    #print('Serializing h5 model to:\n{}'.format(file_output))
    #model.save(file_output)

    info('Test')

    test_loss, test_acc = test(yelp_test_dataset,batch_size, model, criterion, device)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nlp news')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='.data')
    parser.add_argument('-g', '--logs', help='log directory', default='logs')
    parser.add_argument('-o', '--outputs', help='output directory', default='outputs')
    parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=16, type=int)
    parser.add_argument('-r', '--lr', help='learning rate', default=4.0, type=float)
    parser.add_argument('-w', '--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('-u','--dist-url', type=str, help='url used to set up distributed training')
    parser.add_argument('-t','--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('-k','--rank', default=-1, type=int, help='rank of the worker')
    parser.add_argument('-n','--numworkers', default=4, type=int, help='number of workers')
    args = parser.parse_args()

    run = Run.get_context()
    offline = run.id.startswith('OfflineRun')
    print('AML Context: {}'.format(run.id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    distributed = args.world_size >= 2

    args = {
        'run': run,
        'data_path': check_dir(args.data).resolve(),
        'output_path': check_dir(args.outputs).resolve(),
        'log_path': check_dir(args.logs).resolve(),
        'epochs': args.epochs,
        'batch_size': args.batch,
        'learning_rate': args.lr,
        'device': device,
        'distributed': distributed,
        'num_workers': args.numworkers 
    }

    # log output
    if not offline:
        for item in args:
            if item != 'run':
                run.log(item, args[item])

    info('Args')

    for i in args:
        print('{} => {}'.format(i, args[i]))

    main(**args)



# Resources:
# This example is from the [PyTorch Beginner Tutorial](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)

