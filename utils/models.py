class NN(nn.Module):
    """
    Neural network with two fully connected layers.
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim), 
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim))
        
    def forward(self, X):
        return self.net(X)


class CNN_1(nn.Module):
    """
    Convolutional Neural Network, Architecture 1
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64*3*3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,64*3*3)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    

class CNN_2(nn.Module):
    """
    Convolutional Neural Network, Architecture 2
    """

    def __init__(self):
        
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32,64, kernel_size=2)
        self.conv4 = nn.Conv2d(64,64, kernel_size=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        
        x = F.relu(self.conv1(x)) # (26,26,32)
        x = self.max_pool1(x)     # (13,13,32)
        x = F.dropout(x, p=0.3, training=True)
        
        x = F.relu(self.conv2(x)) # (11,11,32)
        x = F.relu(self.conv2(x)) # (9,9,32)
        x = self.max_pool2(x)     # (4,4,32)
        x = F.dropout(x, p=0.3, training=True)
        
        x = F.relu(self.conv3(x)) # (3,3,64)
        x = F.relu(self.conv4(x)) # (2,2,64)
        x = self.max_pool3(x)     # (1,1,64)
        x = F.dropout(x, p=0.3, training=True)
        
        x = self.avg_pool(x)      # (64)
        x = x.view(-1,64)
        x = self.fc(x)
        return x
    

class CNN_3(nn.Module):
    """
    Convolutional Neural Network, Architecture 3
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.avg_pool = nn.AvgPool2d(kernel_size=3)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))

        x = F.dropout(x, p=0.3, training=self.training)
        x = self.avg_pool(x)
        x = x.view(-1,64)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x