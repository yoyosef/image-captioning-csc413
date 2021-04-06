import os
import pandas as pd
from torchtext.data.utils import get_tokenizer
import pickle
from collections import Counter
from torchtext.vocab import Vocab


en_tokenizer = get_tokenizer('spacy', language="en_core_web_sm")


def build_vocab(sentences):
    counter = Counter()
    for sentence in sentences:
        sentence = sentence.lower()
        counter.update(en_tokenizer(sentence))
    return Vocab(counter, specials=['<unk>', '<pad>', '<sos>', '<eos>'])


if __name__ == "__main__":
    data = pd.read_csv("flickr8k/captions.txt")
    captions = data["caption"]
    vocab = build_vocab(captions)
    with open("vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
