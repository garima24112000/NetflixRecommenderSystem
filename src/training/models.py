import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32, mlp_layers=[64,32]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        mlp_input = emb_dim * 2
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_input, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            mlp_input = h
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(mlp_input, 1)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        x = torch.cat([u, v], dim=1)
        x = self.mlp(x)
        x = self.output(x)
        return x.squeeze(1)
