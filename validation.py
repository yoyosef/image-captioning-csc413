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

def get_bleu_score(candidates, references, maxn_gram=4):
    score = bleu_score(candidates, references, max_n=maxn_gram)
    return score

def caption_image(encoderCNN, decoderRNN, image, vocabulary, max_length=50):
    # FROM https://github.com/aladdinpersson/Machine-Learning-Collection/blob/4bd862577ae445852da1c1603ade344d3eb03679/ML/Pytorch/more_advanced/image_captioning/model.py#L49
    # NEED TO CHECK IF IT MAKES SENSE
    result_caption = []

    with torch.no_grad():
        x = encoderCNN(image).unsqueeze(0)
        states = None

        for _ in range(max_length):
            hiddens, states = decoderRNN.lstm(x, states)
            output = decoderRNN.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result_caption.append(predicted.item())
            x = decoderRNN.embedding(predicted).unsqueeze(0)

            if vocabulary.itos[predicted.item()] == "<eos>":
                break

    return [vocabulary.itos[idx] for idx in result_caption][1:-1]


def bulk_caption_image(encoderCNN, decoderRNN, images, vocabulary, max_length=50):
    # FROM https://github.com/aladdinpersson/Machine-Learning-Collection/blob/4bd862577ae445852da1c1603ade344d3eb03679/ML/Pytorch/more_advanced/image_captioning/model.py#L49
    # NEED TO CHECK IF IT MAKES SENSE
    result_caption = [[] for i in range(32)]
    batch_size = 32
    with torch.no_grad():
        x = encoderCNN(images).unsqueeze(1)
        states = None

        for _ in range(max_length):
            
            hiddens, states = decoderRNN.lstm(x, states)
            output = decoderRNN.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            x = decoderRNN.embedding(predicted)
            x = x.unsqueeze(1)

            for i in range(batch_size):
                if vocabulary.itos[predicted[i]] == '<sos>':
                    continue

                if len(result_caption[i]) == 0:
                    result_caption[i].append(predicted[i].item())
                    continue

                if vocabulary.itos[result_caption[i][-1]] != '<eos>':
                    
                    result_caption[i].append(predicted[i].item())

            # if vocabulary.itos[predicted.item()] == "<eos>":
            #     break

    return [[vocabulary.itos[idx] for idx in result_caption[i]][:-1] for i in range(batch_size)]

def yield_data(dataset, root_dir):
    transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])
    grouped = dataset.csv_frame.groupby("image")["caption"].apply(list).reset_index(name="caption")
    # references = grouped["caption"]
    for idx in range(len(grouped)):
        img_name = os.path.join(root_dir, grouped["image"][idx])
        image = Image.open(img_name)
        image = transform(image).unsqueeze(0)
        refs = [en_tokenizer(ref.lower()) for ref in grouped["caption"][idx]]
        yield image, refs

def evaluate_bleu(encoder, decoder, vocabulary, dataset, root_dir="flickr8k/images"):
    """
    This is slow
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval()
    decoder.eval()
    bleu = 0
    count = 0
    for img, refs  in yield_data(dataset, root_dir):
        
        # img = img.to(device)
        cap = caption_image(encoder, decoder, img, vocabulary)
        score = get_bleu_score([cap], [refs])
        
        bleu += score
        count += 1
        if count % 5 == 0:
            print("CAND ", " ".join(cap))
            print("REFS ", "\n".join([" ".join(r) for r in refs]))
            print(bleu/count)

    return bleu/count


