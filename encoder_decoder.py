import torch
import torch.nn as nn
import torchvision.models as models
# ref: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
from torch.nn.utils.rnn import pack_padded_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetAttentionEncoder(nn.Module):
    def __init__(self, embed_size, bn_momentum=0.01):
        super(ResNetAttentionEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), features.size(1), -1)
        features = features.permute(0, 2, 1)
        return features


class ResNetEncoder(nn.Module):
    def __init__(self, embed_size, bn_momentum=0.01):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.batchnorm = nn.BatchNorm1d(embed_size, momentum=bn_momentum)

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.batchnorm(self.linear(features))
        return features


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """ This uses teacher forcing """
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs


class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_size, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(hidden_size, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)

        # (batch_size,num_layers,attemtion_dim)
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))

        attention_scores = self.A(combined_states)  # (batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(
            2)  # (batch_size,num_layers)

        alpha = torch.softmax(attention_scores, dim=1)  # (batch_size,num_layers)

        # (batch_size,num_layers,features_dim)
        attention_weights = features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(
            dim=1)  # (batch_size,num_layers)

        return alpha, attention_weights


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_dim, attention_dim):
        super().__init__()

        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.lstm_cell = nn.LSTMCell(
            embed_size+encoder_dim, hidden_size, bias=True)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)

        self.fcn = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):

        # vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, hidden_size)

        # get the seq length to iterate
        seq_length = len(captions[0])-1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(h)

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def generate_caption(self, features, max_len=20, vocab=None):
        # Inference part
        # Given the image features generate the captions
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, hidden_size)
        num_images = features.shape[0]
        
        # alphas
        alphas = []

        # starting input
        word = torch.tensor(vocab.stoi['<sos>']).view(1, -1).to(device)
        embeds = torch.cat(num_images*[self.embedding(word)])

        result_caption = [[] for i in range(num_images)]
        with torch.no_grad():

            for _ in range(max_len):

                alpha, context = self.attention(features, h)

                # # store the apla score
                alphas.append(alpha.cpu().detach().numpy())
                embeds_0, context = embeds[:, 0], context

                lstm_input = torch.cat((embeds_0, context), dim=1)
                h, c = self.lstm_cell(lstm_input, (h, c))
                output = self.fcn(h)
                output = output.view(batch_size, -1)

                # select the word with most val
                predicted = output.argmax(dim=1)

                for i in range(num_images):
                    if vocab.itos[predicted[i]] == '<sos>':
                        continue

                    if len(result_caption[i]) == 0:
                        result_caption[i].append(predicted[i].item())
                        continue

                    if vocab.itos[result_caption[i][-1]] != '<eos>':

                        result_caption[i].append(predicted[i].item())

                embeds = self.embedding(predicted).unsqueeze(1)

                # if vocabulary.itos[predicted.item()] == "<eos>":
                #     break

        return [[vocab.itos[idx] for idx in result_caption[i]][:-1] for i in range(num_images)], alphas


    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, hidden_size)
        c = self.init_c(mean_encoder_out)
        return h, c
