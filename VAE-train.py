import torch
from VAE import VAE
import torch.optim as optim
import pandas as pd
import argparse
from torch.utils.tensorboard import SummaryWriter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "mps"

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

# Load Data
data = pd.read_json('data/vectorized-data.json')
data = data.values
data = data.reshape(-1)
data = torch.tensor(data.tolist(), dtype=torch.float)
data = data.to(device)

# Shuffle and split data into train and test
torch.manual_seed(10)
perm = torch.randperm(data.size(0))
data = data[perm]
train_data = data[:int(0.8 * data.size(0))]
test_data = data[int(0.8 * data.size(0)):]

# Initialize VAE
model = VAE()
model.to(device)
model.train()

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

writer = SummaryWriter('logs')

best_test_loss = 100000000
# Training loop
for epoch in range(args.epochs):
    for i in range(0, len(train_data), args.batch_size):
        batch = train_data[i:i+args.batch_size]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch)
        loss = model.loss(recon_batch, batch, mu, logvar)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.4f}')
    writer.add_scalar('Train Loss', loss.item(), epoch)
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

    # Test
    if epoch % 5 == 0:
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i in range(0, len(test_data), args.batch_size):
                batch = test_data[i:i+args.batch_size]
                recon_batch, mu, logvar = model(batch)
                test_loss += model.loss(recon_batch, batch, mu, logvar).item()
        test_loss /= len(test_data)
        print(f'Test Loss: {test_loss:.4f}')
        model.train()
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'vae_best.pth')
    writer.add_scalar('Test Loss', test_loss, epoch)