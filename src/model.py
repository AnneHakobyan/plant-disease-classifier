
import timm


def build_model(backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        drop_rate=dropout,
        num_classes=num_classes
    )
    return model


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
