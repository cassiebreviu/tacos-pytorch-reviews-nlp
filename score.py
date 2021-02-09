import time
import json
import torch
import datetime
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.data.utils import ngrams_iterator

from azureml.core.model import Model

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

# globals
model, dictionary, tokenizer, ngrams  = None, None, None, 2
def init():
    global model, dictionary, tokenizer, ngrams
    
    # scoping issue with pickling
    binding = TextSentiment

    try:
        model_files = Model.get_model_path('tacoreviewsmodel')
    except:
        model_files = 'data/output'

    model_path = Path(model_files).resolve()
    print(f'Loading model files from {str(model_path)}')

    model_file = (model_path / 'model.pth').resolve()
    vocab_file = (model_path / 'vocab.data').resolve()

    model = torch.load(str(model_file))
    dictionary = torch.load(str(vocab_file))
    tokenizer = get_tokenizer("basic_english")


def run(raw_data):
    global model, dictionary, tokenizer, ngrams

    prev_time = time.time()
    post = json.loads(raw_data)
    text = post['text']

    with torch.no_grad():
        text = torch.tensor([dictionary[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))

        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)

        payload = {
            'time': str(inference_time.total_seconds()),
            'scores': output[0],
            'rating': output.argmax(1).item() + 1
        }

        print('Input ({}), Prediction ({})'.format(text, payload))

        return payload


if __name__ == '__main__':
    init()
    text = "can't stand eating here"

    print(run(json.dumps({'text': text})))
