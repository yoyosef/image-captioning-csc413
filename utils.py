import matplotlib.pyplot as plt
from torchvision import transforms
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_image(img):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    img = inv_normalize(img).permute(1, 2, 0)
    plt.imshow(img)


def get_caption_attention(encoder, decoder, image, vocab):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(image.to(device))
        caps, alphas = decoder.generate_caption(features, vocab=vocab)
        caps = caps[0]
        caption = ' '.join(caps)
        print(caption)
        plot_image(image.squeeze(0))

    return caps, alphas


def get_caption_lstm(encoder, decoder, image, vocab):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(image.to(device))
        caps = decoder.generate_caption(features, vocab=vocab)
        caps = caps[0]
        caption = ' '.join(caps)
        print(caption)
        plot_image(image.squeeze(0))

    return caps


def plot_attention(img, captions, attention_plot):
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    img = inv_normalize(img).permute(1, 2, 0)
    
    fig = plt.figure(figsize=(15, 15))

    len_captions = len(captions)
    for l in range(len_captions):
        temp_att = attention_plot[l].reshape(7, 7)

        ax = fig.add_subplot(len_captions//2, len_captions//2, l+1)
        ax.set_title(captions[l])
        att_img = ax.imshow(img)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=att_img.get_extent())

    plt.tight_layout()
    plt.show()
