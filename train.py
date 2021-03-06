from data import initialize_loader, Flickr8k
import torch
from torch import nn
from torchvision import transforms
import time
import numpy as np
from encoder_decoder import ResNetEncoder, Decoder, DecoderWithAttention, ResNetAttentionEncoder
from torch.nn.utils.rnn import pack_padded_sequence
import os
import pickle
from validation import evaluate_bleu_batch, validation_bleu1
from pathlib import Path


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    with open("vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    if args.model_type == "attention":
        encoder = ResNetAttentionEncoder(args.embed_size, finetune=args.finetune_attention)
        decoder = DecoderWithAttention(len(
            vocab), args.embed_size, args.hidden_size, args.encoder_dim, args.attention_dim)
    else:
        encoder = ResNetEncoder(args.embed_size)
        decoder = Decoder(len(vocab), args.embed_size, args.hidden_size)

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
    if args.model_type == "attention":
        finetune_params = list(list(encoder.resnet.children())[-1].parameters()) if args.finetune_attention else []
        optimizer = torch.optim.Adam(
            list(decoder.parameters()) + finetune_params, lr=args.learn_rate)
    else:
        optimizer = torch.optim.Adam(list(encoder.linear.parameters(
        )) + list(decoder.parameters()) + list(encoder.batchnorm.parameters()), lr=args.learn_rate)

    encoder.to(device)
    decoder.to(device)
    start = time.time()

    total_step = len(train_loader)
    train_losses = []
    bleu_scores = []
    early_stopping_counter = 0
    best_val_loss = 1e6
    best_bleu = -1

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        losses = []

        for i, (imgs, captions, lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            captions = captions.to(device)
            encoder_output = encoder(imgs)
            if args.model_type == "attention":
                outputs = decoder(encoder_output, captions)[0]
                outputs = outputs.view(-1, outputs.size(2))
                targets = captions[:, 1:]
                targets = targets.reshape(-1)
            else:
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

        bleu = validation_bleu1(encoder, decoder, vocab, val_data, attention=(args.model_type == "attention"))
        bleu_scores.append(bleu)
        print("Epoch [{}/{}], Bleu Score: {}".format(epoch+1, args.epochs, bleu))

        if (epoch+1) % args.save_epoch == 0:
            Path(os.path.join("./", args.model_path)
                 ).mkdir(parents=True, exist_ok=True)
            torch.save(decoder.state_dict(), os.path.join(
                args.model_path, 'decoder-attention-{}.ckpt'.format(epoch+1)))
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-attention-{}.ckpt'.format(epoch+1)))

        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print(
            "Epoch [%d/%d], Loss: %.4f, Time (s): %d"
            % (epoch + 1, args.epochs, avg_loss, time_elapsed)
        )

        if args.early_stopping_metric == "bleu":
            if bleu > best_bleu:
                best_bleu = bleu
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter == args.early_stopping_patience:
                print("Bleu score has not improved in {} epochs, stopping early".format(args.early_stopping_patience))
                print("Obtained highest bleu score of: {}".format(best_bleu))
                return
                
        elif args.early_stopping_metric == "loss":
            val_losses = []
            for i, (imgs, captions, lengths) in enumerate(val_loader):
                optimizer.zero_grad()
                imgs = imgs.to(device)
                captions = captions.to(device)
                encoder_output = encoder(imgs)
                if args.model_type == "attention":
                    outputs = decoder(encoder_output, captions)[0]
                    outputs = outputs.view(-1, outputs.size(2))
                    targets = captions[:, 1:]
                    targets = targets.reshape(-1)
                else:
                    outputs = decoder(encoder_output, captions, lengths)
                    targets = pack_padded_sequence(
                        captions, lengths, batch_first=True, enforce_sorted=False)[0]

                val_loss = criterion(
                    outputs, targets
                )
                val_losses.append(val_loss.data.item())
            avg_val_loss = np.mean(val_losses)    
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            print(
                "Epoch [%d/%d], Val Loss: %.4f"
                % (epoch + 1, args.epochs, avg_val_loss)
            )
            if early_stopping_counter == args.early_stopping_patience:
                print("Validation loss has not improved in {} epochs, stopping early".format(args.early_stopping_patience))
                print("Obtained lowest validation loss of: {}".format(best_val_loss))
                return
