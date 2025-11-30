import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

    def forward(self, u, v):
        return (self.user_emb(u) * self.item_emb(v))


class MLP(nn.Module):
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

    def forward(self, u, v):
        x = torch.cat([self.user_emb(u), self.item_emb(v)], dim=1)
        return self.mlp(x)


class NeuMF(nn.Module):
    """NeuMF: fuse GMF and MLP representations and learn final prediction."""
    def __init__(self, n_users, n_items, emb_dim=32, mlp_layers=[64,32], final_hidden=16):
        super().__init__()
        self.gmf = GMF(n_users, n_items, emb_dim=emb_dim)
        self.mlp = MLP(n_users, n_items, emb_dim=emb_dim, mlp_layers=mlp_layers)

        # size of combined representation: emb_dim (gmf) + last_mlp_dim
        mlp_out = mlp_layers[-1] if len(mlp_layers) > 0 else emb_dim
        combined_dim = emb_dim + mlp_out

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, final_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(final_hidden, 1)
        )

    def forward(self, u, v):
        g = self.gmf(u, v)  # (batch, emb_dim)
        m = self.mlp(u, v)  # (batch, mlp_out)
        x = torch.cat([g, m], dim=1)
        out = self.fusion(x)
        return out.squeeze(1)
