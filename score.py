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
model, dictionary, tokenizer, ngrams, device  = None, None, None, 2, None
def init():
    global model, dictionary, tokenizer, ngrams, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device!')

    try:
        model_files = Model.get_model_path('tacoreviewsmodel')
    except:
        model_files = 'data/output'

    model_path = Path(model_files).resolve()
    print(f'Loading model files from {str(model_path)}')
    
    # load vocabulary
    vocab_file = (model_path / 'vocab.data').resolve()
    dictionary = torch.load(str(vocab_file))

    # load metadata
    meta_file = (model_path / 'metadata.json').resolve()
    with open(str(meta_file)) as f:
        metadata = json.load(f)

    # load model
    model_file = (model_path / 'model.pth').resolve()
    vocab_size = metadata['vocab_size']
    embed_dim = metadata['embed_dim']
    num_class = metadata['num_class']
    model = TextSentiment(vocab_size, embed_dim, num_class)
    model.load_state_dict(torch.load(str(model_file), map_location=device))
    model.to(device)
    
    tokenizer = get_tokenizer("basic_english")


def run(raw_data):
    global model, dictionary, tokenizer, ngrams, device

    prev_time = time.time()
    post = json.loads(raw_data)
    incoming_text = post['text']

    with torch.no_grad():
        text = torch.tensor([dictionary[token]
                            for token in ngrams_iterator(tokenizer(incoming_text), ngrams)])

        output = model(text.to(device), torch.tensor([0]).to(device))

        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)

        payload = {
            'time': str(inference_time.total_seconds()),
            'text': incoming_text,
            'scores': output[0].tolist(),
            'rating': output.argmax(1).item() + 1
        }

        print('Input ({}), Prediction ({})'.format(text, payload))
        return payload

if __name__ == '__main__':
    init()
    text = "can't stand eating here"

    print(run(json.dumps({'text': text})))
