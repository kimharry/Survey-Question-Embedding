import torch
from torch import nn
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def forward(self, recon_x, x, mu, logvar):
        MSE = self.mse_loss(recon_x, x)

        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoding
        # 384 -> 256 -> 128 -> 64 -> 32
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc51 = nn.Linear(32, 32)
        self.fc52 = nn.Linear(32, 32)
        

        # Decoding
        # 32 -> 64 -> 128 -> 256 -> 384
        self.fc6 = nn.Linear(32, 64)
        self.fc7 = nn.Linear(64, 128)
        self.fc8 = nn.Linear(128, 256)
        self.fc9 = nn.Linear(256, 384)
        
        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc51(x), self.fc52(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.relu(self.fc6(z))
        z = self.relu(self.fc7(z))
        z = self.relu(self.fc8(z))
        z = self.relu(self.fc9(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        return Loss().forward(recon_x, x, mu, logvar)