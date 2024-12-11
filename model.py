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

class TransformerModel2(nn.Module):
    '''
    Transformer Model predicting next move to take to solve cube, based on cube state.
    '''
    def __init__(self, input_dim = 54, embed_dim = 128, num_heads = 4, num_layers = 2, output_dim = 12, max_seq = 54):
        super(TransformerModel2, self).__init__()
        self.input_dim = input_dim
        self.embed_dims = embed_dim
        self.max_seq = max_seq
        
        self.embed_token = nn.Linear(6, self.embed_dims) # one hot embedding to latent embedding with embed_dims
        self.pe = nn.Embedding(max_seq, self.embed_dims)

        self.encoder_l = nn.TransformerEncoderLayer(
            d_model  = self.embed_dims,
            nhead = num_heads,
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_l, num_layers = num_layers)

        self.out_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        if x.ndim == 3:
            batch_size, scram_len, in_dim = x.shape
        else:
            batch_size, in_dim = x.shape
            scram_len = 1
        
        assert in_dim == 54


        x = F.one_hot(x, num_classes = 6).to(torch.float)
        x = self.embed_token(x)
        x = x.view(batch_size * scram_len, 54, self.embed_dims)

        positions = torch.arange(54, device = x.device).unsqueeze(0)  
        position_embeddings = self.pe(positions)
        position_embeddings = position_embeddings.expand(batch_size * scram_len, 54, self.embed_dims)
        x += position_embeddings

        x = self.encoder(x)
        if scram_len == 1:
            x = x.view(batch_size, 54, self.embed_dims)
        else:
            x = x.view(batch_size, scram_len, 54, self.embed_dims)
        x = x.mean(dim = -2)
        out = self.out_layer(x)
        return out

class TransformerModel(nn.Module):
    '''
    Transformer Model predicting next move to take to solve cube, based on cube state. (ViT Style, with learnable class embedding)
    '''
    def __init__(self, input_dim=54, embed_dim=256, num_heads=4, num_layers=4, output_dim=12, max_seq=54):
        super(TransformerModel2, self).__init__()
        self.input_dim = input_dim
        self.embed_dims = embed_dim
        self.max_seq = max_seq
        
        self.embed_token = nn.Linear(6, self.embed_dims)  
        self.pe = nn.Embedding(max_seq + 1, self.embed_dims)
        
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))  # Learnable class token

        self.encoder_l = nn.TransformerEncoderLayer(
            d_model=self.embed_dims,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_l, num_layers=num_layers)
        
        self.out_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        if x.ndim == 3:
            batch_size, scram_len, in_dim = x.shape
        else:
            batch_size, in_dim = x.shape
            scram_len = 1
        
        assert in_dim == 54

        x = F.one_hot(x, num_classes=6).to(torch.float)
        x = self.embed_token(x)
        x = x.view(batch_size * scram_len, 54, self.embed_dims)

        positions = torch.arange(54, device=x.device).unsqueeze(0)
        position_embeddings = self.pe(positions)
        position_embeddings = position_embeddings.expand(batch_size * scram_len, 54, self.embed_dims)
        x += position_embeddings

        class_token = self.class_token.expand(batch_size * scram_len, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat([class_token, x], dim=1) 
        
        class_position = torch.tensor([54], device=x.device).expand(1, -1)
        class_position_embedding = self.pe(class_position)
        x[:, 0:, :] += class_position_embedding

        x = self.encoder(x)  
        class_token_output = x[:, 0, :] 
        out = self.out_layer(class_token_output) 
        return out

def get_model():
    model = Model()
    model.to(device)
    model.load_state_dict(torch.load('10000steps.pth'))
    return model


if __name__ == "__main__":
    # Define `model` and load it on device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel().to(device)
    print(model)

    # Test prediction with a fake input tensor
    model.eval()
    sample = torch.randint(0, 6, (10, 26, 54)).to(device)
    print(f"{sample.shape=}, {sample=}")

    with torch.no_grad():
        logits = model(sample)[0, :]
        pdist = nn.functional.softmax(logits, dim=-1)
    print(f"{pdist.shape=}, {pdist=}")