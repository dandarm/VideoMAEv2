import torch.nn as nn
from timm.models import create_model
from timm.models.layers import trunc_normal_


class RegressionHead(nn.Module):
    """Simple regression head for cyclone coordinate prediction."""
    def __init__(self, embed_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        trunc_normal_(self.mlp[1].weight, std=0.02)
        nn.init.constant_(self.mlp[1].bias, 0.)

    def forward(self, x):
        return self.mlp(x)


def create_tracking_model(model_name: str, num_outputs: int = 2, **kwargs) -> nn.Module:
    """Build a model for cyclone tracking starting from a classification checkpoint.

    This function loads the specified transformer architecture with weights
    from ``init_ckpt`` (if provided) while discarding the original
    classification head.  A new :class:`RegressionHead` with ``num_outputs``
    units is attached for coordinate regression.
    """
    extra_kwargs = dict(num_classes=0, **kwargs)
    
    model = create_model(model_name, **extra_kwargs)
    

    model.head = RegressionHead(model.embed_dim, num_outputs)
    return model
