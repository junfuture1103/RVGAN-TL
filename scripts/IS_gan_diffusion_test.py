import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.io import arff

# Load the dataset from the ARFF file
data_path = '../data/arff/yeast1.dat_cleaned.arff'
data, meta = arff.loadarff(data_path)

# Convert to DataFrame
df = pd.DataFrame(data)

# Decode byte strings to strings if necessary
for column in df.select_dtypes([object]).columns:
    if df[column].dtype == object:
        df[column] = df[column].str.decode('utf-8')

# Rename the last column to 'class' if it's not already named as such
if df.columns[-1] != 'class':
    df.rename(columns={df.columns[-1]: 'class'}, inplace=True)

# Encode the class labels
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Separate the positive class data
positive_class_data = df[df['class'] == 1].drop('class', axis=1)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the generator and discriminator networks
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Hyperparameters
input_dim = 100  # Dimension of the input noise vector
output_dim = positive_class_data.shape[1]  # Dimension of the generated data
batch_size = 64
num_epochs = 500
lr = 0.0002

# Prepare data
positive_class_tensor = torch.tensor(positive_class_data.values, dtype=torch.float32)
dataset = TensorDataset(positive_class_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(num_epochs):
    for real_data in dataloader:
        real_data = real_data[0]
        batch_size = real_data.size(0)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real data
        labels_real = torch.ones(batch_size, 1)
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, labels_real)

        # Fake data
        noise = torch.randn(batch_size, input_dim)
        fake_data = generator(noise)
        labels_fake = torch.zeros(batch_size, 1)
        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, labels_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_data)
        loss_G = criterion(output_fake, labels_real)  # Trick discriminator
        loss_G.backward()
        optimizer_G.step()

    # if epoch % 50 == 0:
        print(f'Epoch [{epoch}/{num_epochs}] Loss D: {loss_D.item()}, loss G: {loss_G.item()}')
