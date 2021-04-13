from data import initialize_loader, Flickr8k
import torch
from torch import nn
from torchvision import transforms
import time
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import os
import pickle
from torchtext.data.metrics import bleu_score
from PIL import Image
from torchvision.transforms import ToTensor
from data import *
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')

def get_bleu_score(candidates, references, maxn_gram=4, weights=[.25]*4):
    score = bleu_score(candidates, references, max_n=maxn_gram, weights=weights)
    return score


def bulk_caption_image(encoderCNN, decoderRNN, images, vocabulary, batch_size=32, max_length=50, attention=False):
    # FROM https://github.com/aladdinpersson/Machine-Learning-Collection/blob/4bd862577ae445852da1c1603ade344d3eb03679/ML/Pytorch/more_advanced/image_captioning/model.py#L49
    # NEED TO CHECK IF IT MAKES SENSE
    num_images = images.shape[0]
    result_caption = [[] for i in range(num_images)]
    with torch.no_grad():
        x = encoderCNN(images).unsqueeze(1)
        states = None

        for _ in range(max_length):
            hiddens, states = decoderRNN.lstm(x, states)
            output = decoderRNN.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            x = decoderRNN.embedding(predicted)
            x = x.unsqueeze(1)

            for i in range(num_images):
                if vocabulary.itos[predicted[i]] == '<sos>':
                    continue

                if len(result_caption[i]) == 0:
                    result_caption[i].append(predicted[i].item())
                    continue

                if vocabulary.itos[result_caption[i][-1]] != '<eos>':

                    result_caption[i].append(predicted[i].item())

            # if vocabulary.itos[predicted.item()] == "<eos>":
            #     break

    return [[vocabulary.itos[idx] for idx in result_caption[i]][:-1] for i in range(num_images)]


def yield_batched_data(dataset, batch_size, root_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    grouped = dataset.csv_frame.groupby("image")["caption"].apply(
        list).reset_index(name="caption")
    count = 0
    ref_batch = []
    imgs = []
    for idx in range(len(grouped)):
        count += 1

        img_name = os.path.join(root_dir, grouped["image"][idx])
        image = Image.open(img_name)
        image = transform(image).unsqueeze(0)

        refs = [en_tokenizer(ref.lower()) for ref in grouped["caption"][idx]]

        imgs.append(image)
        ref_batch.append(refs)
        if count == batch_size:
            yield imgs, ref_batch
            count = 0
            imgs = []
            ref_batch = []

    if imgs and ref_batch:
        yield imgs, ref_batch


def evaluate_bleu_batch(encoder, decoder, vocabulary, dataset, batch_size=32, 
                attention=False,
                maxn_gram=2,
                root_dir="flickr8k/images"):
    """
    This is faster
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()
    bleu = 0
    count = 0
    for imgs, refs_batch in yield_batched_data(dataset, batch_size, root_dir):

        imgs = torch.cat(imgs, dim=0)
        imgs = imgs.to(device)
        if not attention:
            with torch.no_grad():
                features = encoder(imgs)
                features = features.unsqueeze(1)
                
                caps = decoder.generate_caption_batch(features, vocab=vocab)
        else:
            with torch.no_grad():
                features = encoder(imgs)
                caps = decoder.generate_caption_batch(features, vocab=vocabulary)
        score = get_bleu_score(caps, refs_batch, maxn_gram=maxn_gram, weights=[1/maxn_gram]*maxn_gram)

        bleu += score*len(caps)
        count += len(caps)

    return bleu/count

def get_captions_and_references(encoder, decoder, vocabulary, dataset, batch_size=2, 
                attention=False,
                root_dir="flickr8k/images"):
    """
    This is faster
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()
    captions = []
    references = []
    for imgs, refs_batch in yield_batched_data(dataset, batch_size, root_dir):
        imgs = torch.cat(imgs, dim=0)
        imgs = imgs.to(device)
        references.extend(refs_batch)
        if not attention:
            with torch.no_grad():
                features = encoder(imgs)
                features = features.unsqueeze(1)
                
                caps = decoder.generate_caption_batch(features, vocab=vocabulary)
                captions.extend(caps)
        else:
            with torch.no_grad():
                features = encoder(imgs)
                caps, alphas = decoder.generate_caption_batch(features, vocab=vocabulary)
                captions.extend(caps)
    return captions, references


def validation(encoders, decoders, vocab, val_data, bleu_max=4, attention=False):
    
    bleu = [[] for i in range(bleu_max)]
    epoch = 0
    for encoder, decoder in zip(encoders, decoders):
        c,r = get_captions_and_references(encoder, decoder, 
                                        vocab, 
                                        val_data, 
                                        attention=attention,
                                        batch_size=128)
        for i in range(bleu_max):
            score = get_bleu_score(c, r, maxn_gram=i + 1, weights=[1/(i+1)]*(i+1))
            bleu[i].append(score)

        epoch += 1

    return bleu

def validation_bleu1(encoder, decoder, vocab, val_data, attention=False):
    c,r = get_captions_and_references(encoder, decoder, 
                                    vocab, 
                                    val_data, 
                                    attention=attention,
                                    batch_size=128)
    n_gram = 1
    score = get_bleu_score(c, r, maxn_gram=n_gram, weights=[1/n_gram]*n_gram)
    return score

def validation_meteor(encoder, decoder, vocab, val_data, attention=False):
    c,r = get_captions_and_references(encoder, decoder, 
                                    vocab, 
                                    val_data, 
                                    attention=attention,
                                    batch_size=128)
    return get_meteor_score(candidates, references)

def get_meteor_score(candidates, references):
    count = 0
    score_total = 0
    for c, r in zip(candidates, references):
        score_total += meteor_score([" ".join(ref) for ref in r], " ".join(c))
        count += 1

    return score_total/count