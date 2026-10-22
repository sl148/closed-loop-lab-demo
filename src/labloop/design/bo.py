"""Bayesian optimization and active learning over the discrete dose space (BoTorch).

The GP models the Bliss excess estimated from measured wells (replicates averaged), on the
features from `DesignSpace.features`. Batch strategies:
- qucb / qei: sequential-greedy discrete optimization in chunks, earlier picks and in-flight
  conditions passed as X_pending (fantasies)
- ts: batch Thompson sampling with pathwise (Matheron) posterior samples; scales to the
  hundreds-of-wells batches of 384-well levels
- al: uncertainty sampling, greedy max posterior std with kriging-believer conditioning
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.pathwise import draw_matheron_paths
from gpytorch.mlls import ExactMarginalLogLikelihood

from labloop.design.base import DesignSpace
from labloop.design.simple import missing_singles
from labloop.evaluate import History
from labloop.types import Condition

warnings.filterwarnings("ignore", module="botorch")
warnings.filterwarnings("ignore", module="linear_operator")
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(__import__('os').environ.get('LABLOOP_TORCH_THREADS', '2')))

MAX_TRAIN = 1200
FIT_MAXITER = 60
MAX_CANDIDATES = 4000
CHUNK = 32
MC_CANDIDATES = 1500
MC_CHUNK = 16  # qUCB / qEI joint batch size; memory grows with its square
MC_EVAL_BATCH = 256
TS_BATCH = 64


def fit_gp(space: DesignSpace, history: History, rng: np.random.Generator, warm: dict | None = None):
    """Returns (model, train_conditions) or (None, []) when there is too little data.
    `warm`: hyperparameters of the previous fit, used as the starting point."""
    est = history.excess_estimates()
    if len(est) < 5:
        return None, []
    conds = sorted(est)
    if len(conds) > MAX_TRAIN:
        top = sorted(conds, key=lambda c: -est[c])[: MAX_TRAIN // 3]
        rest = [c for c in conds if c not in set(top)]
        conds = top + [rest[i] for i in rng.choice(len(rest), MAX_TRAIN - len(top), replace=False)]
    X = torch.tensor(space.features(conds))
    Y = torch.tensor([[est[c]] for c in conds])
    torch.manual_seed(int(rng.integers(2**31)))
    model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
    if warm:
        hyper = {k: v for k, v in warm.items() if not k.startswith("outcome_transform") and "train" not in k}
        model.load_state_dict(hyper, strict=False)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model),
                     optimizer_kwargs={"options": {"maxiter": FIT_MAXITER}})
    return model, conds


def candidate_pool(space: DesignSpace, history: History, rng: np.random.Generator,
                   groups=None) -> list[Condition]:
    pool = [c for c in space.combos if groups is None or c.group() in groups]
    if len(pool) > MAX_CANDIDATES:
        ranked = [c for c, _ in history.ranked_excess()[:200] if groups is None or c.group() in groups]
        idx = rng.choice(len(pool), MAX_CANDIDATES - len(ranked), replace=False)
        pool = list(dict.fromkeys(ranked + [pool[i] for i in idx]))
    return pool


def thompson_batch(model, space: DesignSpace, pool: list[Condition], k: int, pending: list[Condition],
                   rng: np.random.Generator, exclude: set | None = None) -> list[Condition]:
    if k <= 0:
        return []
    if pending:
        # Kriging believer: in-flight points are treated as observed at the posterior mean.
        Xp = torch.tensor(space.features(pending))
        with torch.no_grad():
            mu = model.posterior(Xp).mean
        # BoTorch applies the model's outcome transform to Y inside condition_on_observations.
        model = model.condition_on_observations(Xp, mu)
    Xc = torch.tensor(space.features(pool))
    torch.manual_seed(int(rng.integers(2**31)))
    chosen: list[Condition] = []
    taken = set(exclude or ())
    while len(chosen) < min(k, len(pool)):
        n = min(TS_BATCH, k - len(chosen))
        paths = draw_matheron_paths(model, sample_shape=torch.Size([n]))
        with torch.no_grad():
            # Chunked: one call allocates samples x candidates x (train + fantasy) doubles,
            # which reached 3.8 GB at Level 4. The drawn paths are fixed functions, so
            # chunking does not change the values.
            vals = torch.cat([paths(Xc[i:i + 500]).reshape(n, -1) for i in range(0, len(Xc), 500)], dim=1).numpy()
        for row in vals:  # each sample takes its best candidate not yet taken
            for i in np.argsort(-row):
                if pool[i] not in taken:
                    chosen.append(pool[i])
                    taken.add(pool[i])
                    break
    return chosen


class _GPCache:
    """Refit the GP only when the history changed (several plates per submit share it)."""

    def __init__(self, space: DesignSpace):
        self.space, self.key, self.model, self.warm = space, None, None, None

    def get(self, history: History, rng):
        key = len(history.reads)
        if key != self.key:
            self.model, _ = fit_gp(self.space, history, rng, self.warm)
            self.key = key
            if self.model is not None:
                self.warm = {k: v.clone() for k, v in self.model.state_dict().items()}
        return self.model


class BODesigner:
    """Bayesian optimization of Bliss excess. Before enough combos are measured it seeds
    with singles + random combos."""

    def __init__(self, space: DesignSpace, acq: str = "ts", beta: float = 4.0, **_):
        self.space, self.acq, self.beta = space, acq, beta
        self.name = f"bo-{acq}"
        self.gp = _GPCache(space)

    def _seed(self, history, budget, pending, rng, groups=None):
        from labloop.design.simple import RandomDesigner

        return RandomDesigner(self.space).propose(history, budget, pending, rng, groups)

    def propose(self, history: History, budget: int, pending: list[Condition], rng, groups=None) -> list[Condition]:
        batch = missing_singles(self.space, history, pending, groups)[:budget]
        model = self.gp.get(history, rng)
        if model is None:
            return batch + [c for c in self._seed(history, budget - len(batch), pending + batch, rng, groups)
                            if c not in batch][: budget - len(batch)]
        pool = candidate_pool(self.space, history, rng, groups)
        k = budget - len(batch)
        if self.acq == "ts":
            return batch + thompson_batch(model, self.space, pool, k, list(pending), rng)
        return batch + self._greedy_mc(model, pool, k, list(pending), rng)

    def _greedy_mc(self, model, pool, k, pending, rng) -> list[Condition]:
        if len(pool) > MC_CANDIDATES:  # sequential-greedy MC acquisition is the slow path
            pool = [pool[i] for i in rng.choice(len(pool), MC_CANDIDATES, replace=False)]
        Xc = torch.tensor(self.space.features(pool))
        rows = {tuple(np.round(x, 9)): i for i, x in enumerate(Xc.numpy())}
        chosen: list[Condition] = []
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([64]), seed=int(rng.integers(2**31)))
        train_X = model.train_inputs[0]
        while len(chosen) < min(k, len(pool)):
            q = min(MC_CHUNK, k - len(chosen))
            pend = pending + chosen
            Xp = torch.tensor(self.space.features(pend)) if pend else None
            if self.acq == "qei":
                acqf = qLogNoisyExpectedImprovement(model, X_baseline=train_X, sampler=sampler, X_pending=Xp,
                                                    prune_baseline=True)
            else:
                acqf = qUpperConfidenceBound(model, beta=self.beta, sampler=sampler, X_pending=Xp)
            avoid = torch.tensor(self.space.features(chosen)) if chosen else None
            cand, _ = optimize_acqf_discrete(acqf, q=q, choices=Xc, X_avoid=avoid, unique=True,
                                             max_batch_size=MC_EVAL_BATCH)
            picked = [pool[rows[tuple(np.round(x, 9))]] for x in cand.numpy()]
            if not picked:
                break
            chosen += picked
        return chosen[:k]


class ActiveLearningDesigner:
    """Uncertainty sampling for landscape accuracy: greedy max posterior std, conditioning
    on each chosen chunk at its posterior mean (kriging believer)."""

    name = "al"

    def __init__(self, space: DesignSpace, **_):
        self.space = space
        self.gp = _GPCache(space)

    def propose(self, history: History, budget: int, pending: list[Condition], rng, groups=None) -> list[Condition]:
        batch = missing_singles(self.space, history, pending, groups)[:budget]
        model = self.gp.get(history, rng)
        if model is None:
            return batch + [c for c in BODesigner(self.space)._seed(history, budget - len(batch), pending + batch,
                                                                     rng, groups) if c not in batch][: budget - len(batch)]
        pool = candidate_pool(self.space, history, rng, groups)
        Xc = torch.tensor(self.space.features(pool))
        taken: list[int] = []
        k = budget - len(batch)
        base_pending = torch.tensor(self.space.features(list(pending))) if pending else None
        while len(taken) < min(k, len(pool)):
            # Condition the base model once on everything in flight or already picked (kriging
            # believer); conditioning a chain of fantasy models grows memory without bound.
            parts = [x for x in (base_pending, Xc[taken] if taken else None) if x is not None]
            fantasy = model
            if parts:
                Xp = torch.cat(parts)
                with torch.no_grad():
                    mu = model.posterior(Xp).mean
                fantasy = model.condition_on_observations(Xp, mu)
            with torch.no_grad():
                std = torch.cat([fantasy.posterior(Xc[i:i + 500]).variance.squeeze(-1)
                                 for i in range(0, len(Xc), 500)]).sqrt().numpy()
            std[taken] = -1
            taken += [int(i) for i in np.argsort(-std)[: min(CHUNK, k - len(taken))]]
            del fantasy
        return batch + [pool[i] for i in taken][:k]
