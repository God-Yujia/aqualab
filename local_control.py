"""Output controls for normalized RGB tensors with shape (N, 3, H, W)."""

import torch
import torch.nn.functional as F


RANGE_KNEE = 0.90
GUIDE_RADIUS = 3
GUIDE_EPSILON = 1e-3


def normalize_overflow(prediction: torch.Tensor) -> torch.Tensor:
    """Stretch each out-of-range channel separately, following URanker's tail."""
    low = prediction.amin(dim=(-2, -1), keepdim=True)
    high = prediction.amax(dim=(-2, -1), keepdim=True)
    span = high - low
    needs_stretch = ((low < 0) | (high > 1)) & (span > 0)
    denominator = torch.where(needs_stretch, span, 1.0)
    stretched = (prediction - low) / denominator
    return torch.where(needs_stretch, stretched, prediction.clamp(0, 1))


def _smooth_risk(risk: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    """Close small gaps, then smooth the map along the source image's edges."""
    dilated = F.max_pool2d(risk, kernel_size=3, stride=1, padding=1)
    closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)

    def box_mean(values: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            values, kernel_size=2 * GUIDE_RADIUS + 1, stride=1,
            padding=GUIDE_RADIUS, count_include_pad=False,
        )

    guide_mean = box_mean(guide)
    risk_mean = box_mean(closed)
    variance = (box_mean(guide.square()) - guide_mean.square()).clamp_min(0)
    covariance = box_mean(guide * closed) - guide_mean * risk_mean
    slope = covariance / (variance + GUIDE_EPSILON)
    intercept = risk_mean - slope * guide_mean
    return (box_mean(slope) * guide + box_mean(intercept)).clamp(0, 1)


def limit_local_enhancement(
    source: torch.Tensor, prediction: torch.Tensor, strength: float = 1.0,
    *, smooth: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a bounded RGB result and its per-pixel enhancement coefficient.

    The source is in [0, 1], and prediction is the finite model output before
    clipping. One coefficient scales all three channel changes at each pixel.
    """
    delta = strength * (prediction - source)
    moving = delta != 0
    channel_room = torch.where(delta > 0, 1 - source, source)
    step = torch.where(moving, delta.abs(), 1.0)
    channel_limit = (channel_room / step).masked_fill(~moving, torch.inf)
    limit = channel_limit.amin(dim=1, keepdim=True).clamp(max=1 / RANGE_KNEE)

    # Scale the requested step smoothly as it approaches the RGB cube boundary.
    # The rational shoulder has slope 1 at the knee and tends to the limit.
    onset = RANGE_KNEE * limit
    width = limit - onset
    excess = (1 - onset).clamp_min(0)
    permitted = (onset + width * (excess / (width + excess))).clamp(0, 1)

    fraction = permitted
    if smooth:
        risk = 1 - permitted
        guide = source.amax(dim=1, keepdim=True)  # HSV value of the original image
        smooth_risk = _smooth_risk(risk, guide)
        # Smoothing may lower a pixel's risk. Keep its own limit in that case.
        fraction = torch.minimum(permitted, 1 - smooth_risk)
    return source + fraction * delta, strength * fraction


def apply_output_control(
    source: torch.Tensor, prediction: torch.Tensor, method: str, strength: float,
    *, smooth: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if method == "clip":
        return prediction, None  # The image encoder applies the original clamp.
    if method == "fixed":
        return source.lerp(prediction, strength), None
    if method == "normalize":
        return normalize_overflow(prediction), None
    if method == "local":
        return limit_local_enhancement(source, prediction, strength, smooth=smooth)
    raise ValueError(f"Unknown output control method: {method}")
