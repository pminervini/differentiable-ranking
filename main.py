#!/usr/bin/env python3

import torch
from torch import Tensor

from imle.aimle import aimle
from imle.target import AdaptiveTargetDistribution

def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))

def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]

target_distribution = AdaptiveTargetDistribution(initial_alpha=1.0, initial_beta=0.0)

@aimle(target_distribution=target_distribution)
def differentiable_ranker(weights_batch: Tensor) -> Tensor:
    return rank_normalised(weights_batch)
