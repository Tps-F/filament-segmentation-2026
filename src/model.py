"""Small, pretrained U-Net models for native-resolution filament segmentation.

Inputs are grayscale replicated to RGB, in [0, 1]. Channel zero predicts the
annotated filament pixels; channel one predicts a modestly closed instance mask
and is used only to associate nearby filament pieces during postprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import segmentation_models_pytorch as smp


class FilamentUNet(nn.Module):
    def __init__(
        self,
        encoder: str = "resnet34",
        pretrained: bool = True,
        out_channels: int = 2,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=out_channels,
            decoder_channels=decoder_channels,
        )
        self.register_buffer("input_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1)
        return self.net((image - self.input_mean) / self.input_std)


def build_model(config: dict[str, Any] | None = None, pretrained: bool | None = None) -> FilamentUNet:
    config = config or {}
    if pretrained is None:
        pretrained = bool(config.get("pretrained", True))
    decoder_channels = config.get("decoder_channels", (256, 128, 64, 32, 16))
    if isinstance(decoder_channels, str):
        decoder_channels = tuple(int(x) for x in decoder_channels.split(","))
    return FilamentUNet(
        encoder=config.get("encoder", "resnet34"),
        pretrained=pretrained,
        out_channels=int(config.get("out_channels", 2)),
        decoder_channels=tuple(decoder_channels),
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    use_ema: bool = True,
) -> tuple[FilamentUNet, dict[str, Any]]:
    """Load only architecture + weights; never download encoder weights here."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"], pretrained=False)
    state = checkpoint.get("ema") if use_ema else None
    if state is None:
        state = checkpoint["model"]
    model.load_state_dict(state, strict=True)
    model.input_preprocessing = checkpoint["config"].get("input_preprocessing")
    return model.to(device).eval(), checkpoint


class FilamentLoss(nn.Module):
    """Pixel calibration plus per-positive-example Dice, with a grouping head.

    Empty patches contribute BCE only. This avoids rewarding arbitrary tiny
    foreground responses through Dice smoothing on genuinely empty examples.
    """

    def __init__(self, bridge_weight: float = .3, dice_weight: float = .5) -> None:
        super().__init__()
        self.bridge_weight = bridge_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = logits.float()
        target = target.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((0, 2, 3))
        probs = logits.sigmoid()
        intersection = (probs * target).sum((2, 3))
        mass = target.sum((2, 3))
        dice = (2 * intersection + 1) / (probs.sum((2, 3)) + mass + 1)
        valid = mass > 0
        dice_loss = ((1 - dice) * valid).sum(0) / valid.sum(0).clamp_min(1)
        losses = bce + self.dice_weight * dice_loss
        weights = losses.new_ones(losses.shape)
        if losses.numel() > 1:
            weights[1:] = self.bridge_weight
        loss = (losses * weights).sum() / weights.sum()
        return loss, {"bce": bce[0].detach(), "dice_loss": dice_loss[0].detach(), "bridge_bce": bce[-1].detach()}
