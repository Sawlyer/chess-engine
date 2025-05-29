import torch

class ChessModel(torch.nn.Module):
    def __init__(self, num_classes, n_input_planes=13):
        super().__init__()
        # trunk conv…
        self.conv1 = torch.nn.Conv2d(n_input_planes, 64, 3, padding=1)
        self.bn1   = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn2   = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.bn3   = torch.nn.BatchNorm2d(256)
        self.gap   = torch.nn.AdaptiveAvgPool2d(1)  # (B,256,1,1)→(B,256)

        # policy head
        self.fc_p1 = torch.nn.Linear(256, 256)
        self.fc_p2 = torch.nn.Linear(256, num_classes)

        # value head
        self.fc_v1 = torch.nn.Linear(256, 256)
        self.fc_v2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).view(x.size(0), -1)  # → (B,256)

        # policy
        p = torch.nn.functional.relu(self.fc_p1(x))
        logits = self.fc_p2(p)               # (B, num_classes)

        # value
        v = torch.nn.functional.relu(self.fc_v1(x))
        value = torch.tanh(self.fc_v2(v))    # (B,1) et borné

        return logits, value.squeeze(-1)     # value: (B,)
