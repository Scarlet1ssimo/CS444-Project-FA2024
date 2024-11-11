import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    """
    Linear layer with ReLU and optional BatchNorm
    """

    def __init__(self, input_dim, output_dim, use_bn=True):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with customizable number of LinearBlocks
    """

    def __init__(self, embed_dim, num_layers=2, use_bn=True):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList(
            [LinearBlock(embed_dim, embed_dim, use_bn)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        x += residual  # skip connection
        return x


class Model(nn.Module):
    """
    Customizable model with adjustable linear and residual blocks.
    """

    def __init__(self, input_dim=324, embed_dim=5000, hidden_dim=1000, output_dim=12,
                 num_linear_layers=1, num_residual_blocks=4, use_bn=True):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.embedding = LinearBlock(input_dim, embed_dim, use_bn=use_bn)

        # Add configurable number of linear blocks
        self.linear_layers = nn.ModuleList([
            LinearBlock(embed_dim, hidden_dim, use_bn=use_bn) for _ in range(num_linear_layers)
        ])

        # Add configurable number of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, use_bn=use_bn) for _ in range(num_residual_blocks)
        ])

        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # int indices => float one-hot vectors
        x = F.one_hot(inputs, num_classes=6).to(torch.float)
        x = x.reshape(-1, self.input_dim)

        # Pass through embedding layer
        x = self.embedding(x)

        # Pass through linear layers
        for layer in self.linear_layers:
            x = layer(x)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output layer
        logits = self.output(x)
        return logits


def get_model():
    model = Model()
    model.to(device)
    model.load_state_dict(torch.load('10000steps.pth'))
    return model


if __name__ == "__main__":
    # Define `model` and load it on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    print(model)

    # Test prediction with a fake input tensor
    model.eval()
    sample = torch.randint(0, 6, (1000, 26, 54)).to(device)
    print(f"{sample.shape=}, {sample=}")

    with torch.no_grad():
        logits = model(sample)[0, :]
        pdist = nn.functional.softmax(logits, dim=-1)
    print(f"{pdist.shape=}, {pdist=}")
