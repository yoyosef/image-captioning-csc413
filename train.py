from data import initialize_loader, Flickr8k
import torch
from torch import nn
from torchvision import transforms
import time
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import os
import pickle
from validation import evaluate_bleu

def train(encoder, decoder, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    with open("vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    if args.load_model:
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder.load_state_dict(torch.load(args.decoder_path))

    train_data = Flickr8k(csv_file="flickr8k/train.csv",
                          root_dir="flickr8k/images", vocab=vocab, transform=transform)
    train_loader = initialize_loader(train_data, batch_size=args.batch_size)

    val_data = Flickr8k(csv_file="flickr8k/val.csv",
                        root_dir="flickr8k/images", vocab=vocab, transform=transform)
    val_loader = initialize_loader(val_data, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = torch.optim.Adam(list(encoder.linear.parameters(
    )) + list(decoder.parameters()) + list(encoder.batchnorm.parameters()), lr=args.learn_rate)

    encoder.to(device)
    decoder.to(device)
    start = time.time()
    for param in encoder.resnet.parameters():
        param.requires_grad = False

    total_step = len(train_loader)
    train_losses = []
    for epoch in range(args.epochs):
        losses = []
        for i, (imgs, captions, lengths) in enumerate(train_loader):

            bleu = evaluate_bleu(encoder, decoder, vocab, val_data)
            print("Bleu {}".format(bleu))
            optimizer.zero_grad()
            imgs = imgs.to(device)
            captions = captions.to(device)
            encoder_output = encoder(imgs)
            outputs = decoder(encoder_output, captions, lengths)
            targets = pack_padded_sequence(
                captions, lengths, batch_first=True, enforce_sorted=False)[0]

            loss = criterion(
                outputs, targets
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, args.epochs, i, total_step, loss.item()))

            # if (i+1) % args.save_step == 0:
            #     torch.save(decoder.state_dict(), os.path.join(
            #         args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            #     torch.save(encoder.state_dict(), os.path.join(
            #         args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))

        if (epoch+1) % args.save_epoch == 0:
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-{}.ckpt'.format(epoch+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-{}.ckpt'.format(epoch+1)))

        # avg_loss = np.mean(losses)
        # train_losses.append(avg_loss)
        # time_elapsed = time.time() - start
        # print(
        #     "Epoch [%d/%d], Loss: %.4f, Time (s): %d"
        #     % (epoch + 1, args.epochs, avg_loss, time_elapsed)
        # )
