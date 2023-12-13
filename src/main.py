from model import RNN
from data import IMDB
from parameters import TRAIN_PATH, TEST_PATH
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


trainset, testset = IMDB(TRAIN_PATH), IMDB(TEST_PATH)
trainloader = DataLoader(trainset, batch_size=4, num_workers=2, shuffle=True)
testloader = DataLoader(testset, batch_size=4, num_workers=2, shuffle=False)


criterion = nn.CrossEntropyLoss()
net = RNN()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        inputs, labels = data
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:  # выводим каждую 2000 мини-батчу
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print("Over")
