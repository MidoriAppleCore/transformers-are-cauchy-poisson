#!/usr/bin/env python3
"""
MNIST: Kronecker ``A^T X B`` → complex heads → shared Möbius (or degree-2 rational) warp →
multi-prototype Cauchy kernel classifier (no hidden MLP).

**Gauge / optimization (default):** denominator fixed ``d = 1`` (Möbius) or ``F = 1`` (degree-2); Cauchy pole
cloud centered at the origin each forward pass (removes joint translation drift vs ``b``). Optional
``--optimizer riemannian`` uses ``geoopt``: unit directions for ``(a,b,c)`` on ``S^1 ⊂ ℝ²`` with separate
positive magnitudes, optimized via ``RiemannianAdam`` + ``AdamW`` on everything else (degree **1** only).

Example::

    python mnist_cauchy_poisson.py --kron-rank 6 --kron-prototypes 3 --epochs 50
    python mnist_cauchy_poisson.py --optimizer riemannian   # pip install geoopt
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def kron_AB_x(A: torch.Tensor, B: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """``A,B`` shape ``(S, r)``, ``x2`` ``(B, S, S)`` → ``(B, r, r)``."""
    mid = torch.einsum("rk,bkl->brl", A.T, x2)
    return torch.einsum("brl,lj->brj", mid, B)


class KroneckerMobiusLibrary(nn.Module):
    """
    Kronecker ``A^T X B`` (or bilinear) → ``r²/2`` complex heads, shared **Möbius** or degree-2 rational map.

    Per digit ``k``, head ``h``: ``M`` Cauchy prototypes; scores via ``logsumexp`` over prototypes,
    sum over heads → ``log_softmax``.

    **Gauge:** ``d = 1`` fixed (Möbius) and ``F = 1`` fixed (degree-2 denominator tail).
    **Poles:** global centroid subtracted each forward pass so only relative layout affects kernels.
    **Optional** ``mobius_sphere``: ``a,b,c`` directions on ``S^1``, magnitudes via ``softplus(raw)``.
    """

    def __init__(
        self,
        rank: int = 6,
        n_prototypes: int = 3,
        eps: float = 1e-4,
        use_pool: bool = False,
        bilinear: bool = False,
        mobius_degree: int = 1,
        mobius_sphere: bool = False,
    ):
        super().__init__()
        if rank < 2 or rank > 10:
            raise ValueError("rank must be in [2, 10]")
        if (rank * rank) % 2 != 0:
            raise ValueError("rank^2 must be even")
        if n_prototypes < 1 or n_prototypes > 8:
            raise ValueError("n_prototypes must be in [1, 8]")
        if mobius_degree not in (1, 2):
            raise ValueError("mobius_degree must be 1 or 2")
        if mobius_sphere and mobius_degree != 1:
            raise ValueError("mobius_sphere requires mobius_degree == 1")

        self.rank = rank
        self.n_heads = (rank * rank) // 2
        self.n_prototypes = n_prototypes
        self.eps = eps
        self.use_pool = use_pool
        self.bilinear = bilinear
        self.mobius_degree = mobius_degree
        self.mobius_sphere = mobius_sphere
        self.grid_side = 14 if use_pool else 28

        self.register_buffer("_mobius_d_r", torch.tensor(1.0))
        self.register_buffer("_mobius_d_i", torch.tensor(0.0))

        if bilinear:
            self.A1 = nn.Parameter(torch.randn(self.grid_side, rank) * 0.03)
            self.B1 = nn.Parameter(torch.randn(self.grid_side, rank) * 0.03)
            self.A2 = nn.Parameter(torch.randn(self.grid_side, rank) * 0.03)
            self.B2 = nn.Parameter(torch.randn(self.grid_side, rank) * 0.03)
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        else:
            self.A = nn.Parameter(torch.randn(self.grid_side, rank) * 0.03)
            self.B = nn.Parameter(torch.randn(self.grid_side, rank) * 0.03)
            self.register_parameter("A1", None)
            self.register_parameter("B1", None)
            self.register_parameter("A2", None)
            self.register_parameter("B2", None)

        if mobius_degree == 1:
            if mobius_sphere:
                from geoopt import ManifoldParameter, Sphere

                S = Sphere()
                self.a_u = ManifoldParameter(torch.tensor([1.0, 0.0]), manifold=S)
                self.b_u = ManifoldParameter(torch.tensor([1.0, 0.0]), manifold=S)
                self.c_u = ManifoldParameter(torch.tensor([1.0, 0.0]), manifold=S)
                sm1 = math.log(math.expm1(1.0))
                self.a_mag_raw = nn.Parameter(torch.tensor(sm1))
                self.b_mag_raw = nn.Parameter(torch.tensor(-12.0))
                self.c_mag_raw = nn.Parameter(torch.tensor(-12.0))
                for nm in (
                    "a_r",
                    "a_i",
                    "b_r",
                    "b_i",
                    "c_r",
                    "c_i",
                ):
                    self.register_parameter(nm, None)
            else:
                self.a_r = nn.Parameter(torch.tensor(1.0))
                self.a_i = nn.Parameter(torch.tensor(0.0))
                self.b_r = nn.Parameter(torch.tensor(0.0))
                self.b_i = nn.Parameter(torch.tensor(0.0))
                self.c_r = nn.Parameter(torch.tensor(0.0))
                self.c_i = nn.Parameter(torch.tensor(0.0))
                self.register_parameter("a_u", None)
                self.register_parameter("b_u", None)
                self.register_parameter("c_u", None)
                self.register_parameter("a_mag_raw", None)
                self.register_parameter("b_mag_raw", None)
                self.register_parameter("c_mag_raw", None)
            for nm in (
                "rq_a_r",
                "rq_a_i",
                "rq_b_r",
                "rq_b_i",
                "rq_c_r",
                "rq_c_i",
                "rq_d_r",
                "rq_d_i",
                "rq_e_r",
                "rq_e_i",
            ):
                self.register_parameter(nm, None)
        else:
            for nm in (
                "a_r",
                "a_i",
                "b_r",
                "b_i",
                "c_r",
                "c_i",
                "a_u",
                "b_u",
                "c_u",
                "a_mag_raw",
                "b_mag_raw",
                "c_mag_raw",
            ):
                self.register_parameter(nm, None)
            z = torch.tensor(0.0)
            o = torch.tensor(1.0)
            sm = torch.randn(10) * 0.03
            self.rq_a_r = nn.Parameter(z + sm[0])
            self.rq_a_i = nn.Parameter(z + sm[1])
            self.rq_b_r = nn.Parameter(o + sm[2])
            self.rq_b_i = nn.Parameter(z + sm[3])
            self.rq_c_r = nn.Parameter(z + sm[4])
            self.rq_c_i = nn.Parameter(z + sm[5])
            self.rq_d_r = nn.Parameter(z + sm[6])
            self.rq_d_i = nn.Parameter(z + sm[7])
            self.rq_e_r = nn.Parameter(z + sm[8])
            self.rq_e_i = nn.Parameter(z + sm[9])
            self.register_buffer("_rq_f_r", torch.tensor(1.0))
            self.register_buffer("_rq_f_i", torch.tensor(0.0))

        self.pole_xy = nn.Parameter(
            torch.randn(10, self.n_heads, n_prototypes, 2) * 0.1
        )
        self.raw_y = nn.Parameter(
            torch.full((10, self.n_heads, n_prototypes), math.log(math.expm1(0.5)))
        )

    @staticmethod
    def _complex_from_sphere(u: torch.Tensor, mag_raw: torch.Tensor) -> torch.Tensor:
        m = F.softplus(mag_raw)
        return m * torch.complex(u[0], u[1])

    def mobius_batch(self, z: torch.Tensor) -> torch.Tensor:
        if self.mobius_degree == 1:
            d = torch.complex(self._mobius_d_r, self._mobius_d_i)
            if self.mobius_sphere:
                a = self._complex_from_sphere(self.a_u, self.a_mag_raw)
                b = self._complex_from_sphere(self.b_u, self.b_mag_raw)
                c = self._complex_from_sphere(self.c_u, self.c_mag_raw)
            else:
                a = torch.complex(self.a_r, self.a_i)
                b = torch.complex(self.b_r, self.b_i)
                c = torch.complex(self.c_r, self.c_i)
            den = c * z + d
        else:
            qa = torch.complex(self.rq_a_r, self.rq_a_i)
            qb = torch.complex(self.rq_b_r, self.rq_b_i)
            qc = torch.complex(self.rq_c_r, self.rq_c_i)
            qd = torch.complex(self.rq_d_r, self.rq_d_i)
            qe = torch.complex(self.rq_e_r, self.rq_e_i)
            qf = torch.complex(self._rq_f_r, self._rq_f_i)
            z2 = z * z
            num = qa * z2 + qb * z + qc
            den = qd * z2 + qe * z + qf
            den = torch.where(
                den.abs() < self.eps,
                den / (den.abs() + 1e-12) * self.eps,
                den,
            )
            q_c = num / den
            return torch.stack((q_c.real, q_c.imag), dim=-1)

        den = torch.where(
            den.abs() < self.eps,
            den / (den.abs() + 1e-12) * self.eps,
            den,
        )
        q_c = (a * z + b) / den
        return torch.stack((q_c.real, q_c.imag), dim=-1)

    def encode_queries(self, x: torch.Tensor) -> torch.Tensor:
        """Flattened batch ``(B, 784)`` → query points ``q ∈ ℝ²`` per head, shape ``(B, n_heads, 2)``."""
        x2 = x.view(-1, 28, 28)
        if self.use_pool:
            x2 = F.max_pool2d(x2.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if self.bilinear:
            m1 = kron_AB_x(self.A1, self.B1, x2)
            m2 = kron_AB_x(self.A2, self.B2, x2)
            m = m1 * m2
        else:
            m = kron_AB_x(self.A, self.B, x2)
        v = m.reshape(m.size(0), self.rank * self.rank)
        z = torch.complex(v[:, 0::2], v[:, 1::2])
        return self.mobius_batch(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.encode_queries(x)

        y = F.softplus(self.raw_y) + self.eps
        poles = self.pole_xy
        mu = poles.mean(dim=(0, 1, 2), keepdim=True)
        poles_c = poles - mu

        q_e = q.unsqueeze(1).unsqueeze(3)
        p_e = poles_c.unsqueeze(0)
        diff = q_e - p_e
        r2 = (diff * diff).sum(-1).clamp(min=self.eps ** 2)
        log_kernel = torch.log(y.unsqueeze(0).clamp(min=1e-8)) - torch.log(r2)
        log_kernel = torch.clamp(log_kernel, max=25.0)
        per_k_head = torch.logsumexp(log_kernel, dim=-1)
        logits = per_k_head.sum(dim=-1)
        return F.log_softmax(logits, dim=-1)


class _OptimizerBundle:
    """Single ``zero_grad`` / ``step`` over one or two underlying PyTorch / geoopt optimizers."""

    __slots__ = ("opts",)

    def __init__(self, *opts: torch.optim.Optimizer):
        self.opts = opts

    def zero_grad(self, set_to_none: bool = False) -> None:
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.opts:
            o.step()


def build_training_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    name: str,
) -> _OptimizerBundle:
    if name == "adamw":
        return _OptimizerBundle(
            torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        )
    if name != "riemannian":
        raise ValueError(f"unknown optimizer {name!r}")

    try:
        from geoopt import ManifoldParameter
        from geoopt.optim import RiemannianAdam
    except ImportError as e:
        raise SystemExit(
            "pip install geoopt   # required for --optimizer riemannian"
        ) from e

    mani: list[torch.nn.Parameter] = []
    euclid: list[torch.nn.Parameter] = []
    for p in model.parameters():
        if isinstance(p, ManifoldParameter):
            mani.append(p)
        else:
            euclid.append(p)
    if not mani:
        raise SystemExit(
            "--optimizer riemannian requires ManifoldParameter tensors "
            "(train with --kron-mobius-degree 1 and mobius_sphere enabled)."
        )
    opt_r = RiemannianAdam(mani, lr=lr, stabilize=True)
    opt_e = torch.optim.AdamW(euclid, lr=lr, weight_decay=weight_decay)
    return _OptimizerBundle(opt_r, opt_e)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: Any,
    device: torch.device,
    epoch: int,
    n_epochs: int,
    quiet: bool,
    grad_clip: Optional[float] = None,
) -> float:
    model.train()
    total, correct = 0, 0
    pbar = tqdm(
        loader,
        desc=f"MNIST train {epoch}/{n_epochs}",
        leave=False,
        file=sys.stderr,
        mininterval=0.12,
        dynamic_ncols=True,
        disable=quiet,
    )
    for bi, (x, y) in enumerate(pbar):
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        opt.zero_grad()
        logp = model(x)
        loss = F.nll_loss(logp, y)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += y.size(0)
        correct += (logp.argmax(-1) == y).sum().item()
        if not quiet:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / max(total, 1):.4f}")
        if bi == 0 and epoch == 1:
            print(
                f"[ep1 batch0] loss={loss.item():.4f} (download/extract + first CUDA step OK)",
                flush=True,
            )
    return correct / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)
        logp = model(x)
        total += y.size(0)
        correct += (logp.argmax(-1) == y).sum().item()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="MNIST — Kronecker + Möbius + multi-prototype Cauchy library",
    )
    p.add_argument(
        "--kron-rank",
        type=int,
        default=6,
        help="separable rank r (grid × r); need r² even (e.g. 4, 6, 8, 10)",
    )
    p.add_argument(
        "--kron-prototypes",
        type=int,
        default=3,
        help="M Cauchy prototypes per (digit, head); soft-OR via logsumexp",
    )
    p.add_argument(
        "--kron-pool2",
        action="store_true",
        help="fixed 2×2 max-pool → 14×14 Kronecker grid (0 extra params)",
    )
    p.add_argument(
        "--kron-bilinear",
        action="store_true",
        help="(A₁^T X B₁) ⊙ (A₂^T X B₂) (~2× Kronecker params)",
    )
    p.add_argument(
        "--kron-mobius-degree",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = Möbius; 2 = rational (A z²+B z+C)/(D z²+E z+F)",
    )
    p.add_argument(
        "--optimizer",
        choices=("adamw", "riemannian"),
        default="adamw",
        help="adamw = AdamW everything; riemannian = geoopt RiemannianAdam on Möbius dirs + AdamW rest (degree 1)",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="if >0, clip grad norm; 0 disables",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--test-every",
        type=int,
        default=1,
        help="evaluate test every N epochs",
    )
    p.add_argument("--quiet", action="store_true", help="disable tqdm batch bar")
    p.add_argument(
        "--checkpoint-out",
        type=str,
        default="",
        help="if set, save model state_dict to this path after training (for mnist_kron_visualize.py)",
    )
    args = p.parse_args()

    kr = int(args.kron_rank)
    if (kr * kr) % 2 != 0:
        raise SystemExit("--kron-rank must yield even r² (e.g. 4, 6, 8, 10)")
    mp = max(1, min(int(args.kron_prototypes), 8))

    optimizer_kind = str(args.optimizer)
    if optimizer_kind == "riemannian" and int(args.kron_mobius_degree) != 1:
        print(
            "[train] --optimizer riemannian needs --kron-mobius-degree 1; using adamw",
            flush=True,
        )
        optimizer_kind = "adamw"
    mobius_sphere = optimizer_kind == "riemannian"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([transforms.ToTensor()])
    print(
        "[init] MNIST: download/extract may pause here (first run only) …",
        flush=True,
    )
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tf)
    print("[init] train set ready; loading test set …", flush=True)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    print(
        f"[init] {len(train_ds)} train / {len(test_ds)} test  |  batch={args.batch}  "
        f"|  batches/epoch={math.ceil(len(train_ds) / args.batch)}",
        flush=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    model = KroneckerMobiusLibrary(
        rank=kr,
        n_prototypes=mp,
        use_pool=bool(args.kron_pool2),
        bilinear=bool(args.kron_bilinear),
        mobius_degree=int(args.kron_mobius_degree),
        mobius_sphere=mobius_sphere,
    ).to(device)

    print(
        f"[device] {device}  |  kron-rank={kr}  kron-prototypes={mp}"
        f"  pool2={args.kron_pool2}  bilinear={args.kron_bilinear}"
        f"  kron-mobius-deg={args.kron_mobius_degree}"
        f"  optimizer={optimizer_kind}"
        f"  gauge=d_fixed centroid_poles mobius_sphere={mobius_sphere}"
        f"  |  trainable params: {count_parameters(model):,}",
        flush=True,
    )

    opt = build_training_optimizer(model, args.lr, 1e-4, optimizer_kind)

    best_test = 0.0
    test_every = max(1, args.test_every)
    gclip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None
    for ep in range(1, args.epochs + 1):
        tr_acc = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            epoch=ep,
            n_epochs=args.epochs,
            quiet=args.quiet,
            grad_clip=gclip,
        )
        if ep % test_every == 0 or ep == args.epochs:
            te_acc = evaluate(model, test_loader, device)
            best_test = max(best_test, te_acc)
            print(
                f"epoch {ep:3d}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}  best_test={best_test:.4f}",
                flush=True,
            )
        else:
            print(
                f"epoch {ep:3d}  train_acc={tr_acc:.4f}  test_acc=(skip)  best_test={best_test:.4f}",
                flush=True,
            )

    print(f"\nDone. Best test accuracy: {best_test:.4f}", flush=True)

    if args.checkpoint_out:
        path = args.checkpoint_out
        torch.save(model.state_dict(), path)
        print(f"[checkpoint] saved {path}", flush=True)


if __name__ == "__main__":
    main()
