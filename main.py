#!/usr/bin/env python3

import torch
from torch import nn, Tensor

from imle.aimle import aimle
from imle.target import AdaptiveTargetDistribution, TargetDistribution

def rank(seq: Tensor) -> Tensor:
    res = torch.argsort(torch.argsort(seq, dim=1, descending=True)) + 1
    return res.float()

# Adaptive Implicit MLE (https://arxiv.org/abs/2209.04862, AAAI 2023)
target_distribution = AdaptiveTargetDistribution(beta_update_step=1e-2)
# Implicit MLE (https://arxiv.org/abs/2106.01798, NeurIPS 2021)
# target_distribution = TargetDistribution(alpha=1.0, beta=100.0)

@aimle(target_distribution=target_distribution)
def differentiable_ranker(weights_batch: Tensor) -> Tensor:
    return rank(weights_batch)

class MeanReciprocalRank:
    def __init__(self):
      pass

    def __call__(self,
                 input: Tensor,
                 target: Tensor) -> Tensor:
        ranks_2d = differentiable_ranker(input)
        batch_size = ranks_2d.shape[0]
        ranks = ranks_2d[torch.arange(batch_size), target]
        return torch.mean(1.0 / ranks)

mrr_f = MeanReciprocalRank()

nb_instances = 32
nb_elements = 64

test_input = nn.Parameter(torch.randn(nb_instances, nb_elements) * 1e-3, requires_grad=True)
test_target = torch.randint(high=nb_elements, size=(nb_instances,))

optimiser = torch.optim.Adagrad([test_input], lr=1.0)

for i in range(1000):
    mrr = mrr_f(test_input, test_target)
    loss = 1.0 - mrr
    loss.backward()

    if i % 10 == 0:
        print(f'Iteration {i}, Training MRR: {mrr.item()} (max is 1.0)')
    optimiser.step()
    optimiser.zero_grad()