import torch
from torchmetrics import SpectralAngleMapper, StructuralSimilarityIndexMeasure


ssim_fn = StructuralSimilarityIndexMeasure(return_full_image = True)
def sam_fn(pred, target):
    '''
    pred, target: [c, w, h]
    '''
    pred, target = pred.squeeze(), target.squeeze()
    up = torch.sum((target*pred), dim = 0)   # [w, h]
    down1 = torch.sum((target**2), dim = 0).sqrt()
    down2 = torch.sum((pred**2), dim = 0).sqrt()

    map = torch.arccos(up / (down1 * down2))
    score = torch.mean(map[~torch.isnan(map)])
    map[torch.isnan(map)] = 0
    return score, map