import torch.nn as nn

class DSEModel(nn.Module):
    def __init__(self, num_classes):
        super(DSEModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.rnn = nn.LSTM(256, 256, batch_first=True, bidirectional=True)
        self.embedding = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.BatchNorm1d(num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(3)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.embedding(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class SpeechDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.speech_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.speech_files)

    def __getitem__(self, idx):
        speech_file = self.speech_files[idx]
        speech, sr = torchaudio.load(os.path.join(self.data_dir, speech_file))
        speech = speech.mean(dim=0, keepdim=True)
        spectrogram = torchaudio.transforms.MelSpectrogram(sr)(speech)
        label = int(speech_file.split("_")[0])
        return spectrogram, label

speech_dataset = SpeechDataset(data_dir)
speech_dataloader = DataLoader(speech_dataset, batch_size=batch_size, shuffle=True)

model = DSEModel(num_classes)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

for epoch in range(num_epochs):
    for i, (spectrogram1, label1) in enumerate(speech_dataloader):
        spectrogram1 = spectrogram1.to(device)
        label1 = label1.to(device)
        
        optimizer.zero_grad()

        output1 = model(spectrogram1)

        spectrogram2, label2 = get_random_sample(speech_dataset, label1)
        spectrogram2 = spectrogram2.to(device)
        label2 = label2.to(device)

        output2 = model(spectrogram2)

        loss = criterion(output1, output2, label1 == label2)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print("Epoch: {}, Iteration: {}, Loss: {}".format(epoch, i, loss.item()))

    scheduler.step()
    torch.save(model.state_dict(), "model_weights.pt")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for spectrogram, label in test_dataloader:
        spectrogram = spectrogram.to(device)
        label = label.to(device)

        output = model(spectrogram)

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print("Test Accuracy: {}%".format(accuracy))
import matplotlib.pyplot as plt

# assuming you have stored the model's training history in a variable called 'history'
# plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# plot the training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.show()