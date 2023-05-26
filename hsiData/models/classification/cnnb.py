import torch 

class model(torch.nn.Module):
    def __init__(self, input_channels, n_classes, dropout=False):
        super(model, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(input_channels, 2048)
        self.fc2 = torch.nn.Linear(2048, 4096)
        self.fc3 = torch.nn.Linear(4096, 2048)
        self.fc4 = torch.nn.Linear(2048, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x
