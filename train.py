from data import initialize_loader

def train(encoder, decoder, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    train_data = Flickr8k(csv_file="flickr8k/train.csv", root_dir="flickr8k/images", transform=transform)
    train_loader = initialize_loader(dataset, batch_size=args.batch_size)

    val_data = Flickr8k(csv_file="flickr8k/val.csv", root_dir="flickr8k/images", transform=transform)
    val_loader = initialize_loader(dataset, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<PAD>"])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learn_rate)

    encoder.to(device)
    decoder.to(device)
    start = time.time()

    train_losses = []
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        losses = []
        for i, (imgs, captions, lengths) in enumerate(train_loader):
            optimizer.zero_grad()

            encoder_output = encoder(imgs)
            outputs = decoder(encoder_output, captions, lengths)

            loss = criterion(outputs, captions, batch_size=args.batch_size)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print(
            "Epoch [%d/%d], Loss: %.4f, Time (s): %d"
            % (epoch + 1, args.epochs, avg_loss, time_elapsed)
        )
   