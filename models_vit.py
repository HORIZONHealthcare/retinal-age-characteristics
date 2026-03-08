import timm
import torch
import torch.nn as nn



class RETFound_dinov2_MM(nn.Module):
    def __init__(self, n_modalities=1, enable_projector=False):
        super().__init__()
        self.backbone_list = nn.ModuleList([
            timm.create_model(
                'vit_large_patch14_dinov2.lvd142m',
                pretrained=True,
                img_size=224
            )
            for _ in range(n_modalities)
        ])
        embed_dim = 1024
        self.enable_projector = enable_projector
        if self.enable_projector:
            hidden_dim = embed_dim // 2
            self.projector_list = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(n_modalities)
            ])
        else: hidden_dim = embed_dim
        self.linear = PredictionHead(embed_dim=hidden_dim * n_modalities)

    def freeze_backbones(self):
        for backbone in self.backbone_list:
            for param in backbone.parameters():
                param.requires_grad = False

    def get_head_parameters(self):
        p_list = []
        if self.enable_projector:
            for projector in self.projector_list:
                p_list.extend(projector.parameters())
        p_list.extend(self.linear.parameters())
        return p_list

    def forward(self, x):
        n_modalities = x.shape[1]
        x_list = []
        for i in range(n_modalities):
            x_ = self.backbone_list[i].forward_features(x[:, i])
            x_ = x_[:, 1:, :].mean(dim=1)
            x_ = self.backbone_list[i].fc_norm(x_)
            if self.enable_projector:
                x_ = self.projector_list[i](x_)
            x_list.append(x_)
        x = torch.cat(x_list, dim=1)
        return self.linear(x)


def RETFound_dinov2():
    backbone = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224
    )
    backbone.head_mi = PredictionHead(embed_dim=1024)

    def forward(x):
        x = backbone.forward_features(x)
        x = x[:, 1:, :].mean(dim=1)
        x = backbone.fc_norm(x)
        return backbone.head_mi(x)

    backbone.forward = forward
    return backbone



class PredictionHead(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.double_linear = nn.Sequential(
            nn.Linear(embed_dim, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, x):
        return self.double_linear(x)
