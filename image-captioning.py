import torch
import torch.nn as nn
import torchvision.models as models
# ref: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning 

class AlexNetEncoder(nn.Module):
    def __init__(self, bn_momentum=0.01):
        super(AlexNetEncoder, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        modules = list(alexnet.children())[:-1]
        self.alexnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.batchnorm = nn.BatchNorm1d(embed_size, momentum=bn_momentum)

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        features = self.resnet(x)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.batchnorm(self.linear(features))
        return features

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """ This uses teacher forcing """
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs