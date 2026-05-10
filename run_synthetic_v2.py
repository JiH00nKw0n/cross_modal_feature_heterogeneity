"""Synthetic Theorem 2 experiment runner (HF Trainer-based).

Usage:
    python run_synthetic_v2.py --config configs/synthetic/alpha_sweep.yaml
    python run_synthetic_v2.py --config configs/synthetic/lambda_sweep.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import TrainingArguments
from transformers.data.data_collator import default_data_collator

from src.configs.experiment import ExperimentConfig, MethodConfig
from src.datasets.synthetic_paired_builder import SyntheticPairedBuilder
from src.datasets.synthetic_pairs import SyntheticPairedDataset
from src.metrics.evaluate import aggregate, evaluate_method
from src.models.configuration_sae import TopKSAEConfig, TwoSidedTopKSAEConfig
from src.models.modeling_sae import TopKSAE, TwoSidedTopKSAE
from src.runners.synthetic_trainers import (
    OursTrainer,
    PairedAuxTrainer,
    PermutationCallback,
    SingleReconTrainer,
    TwoReconTrainer,
)
from src.training.callbacks import SAENormCallback
from src.training.losses import (
    group_sparse_loss,
    iso_alignment_penalty,
    trace_alignment_loss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def _make_training_args(
    cfg: ExperimentConfig,
    output_dir: str,
    seed: int,
) -> TrainingArguments:
    t = cfg.training
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.num_epochs,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        learning_rate=t.lr,
        lr_scheduler_type="constant",
        dataloader_drop_last=True,
        weight_decay=t.weight_decay,
        max_grad_norm=t.max_grad_norm,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        fp16=False,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )


# ------------------------------------------------------------------
# Per-method train+eval
# ------------------------------------------------------------------


def _save_params(
    sae_i: TopKSAE,
    sae_t: TopKSAE,
    builder: SyntheticPairedBuilder,
    alpha: float,
    seed: int,
    method_id: str,
    run_dir: Path,
) -> None:
    dump_dir = run_dir / "params"
    dump_dir.mkdir(parents=True, exist_ok=True)
    mid_safe = method_id.replace("::", "__").replace("/", "_")
    fn = dump_dir / f"alpha{alpha:.2f}_seed{seed}_{mid_safe}.npz"
    np.savez_compressed(
        fn,
        w_dec_img=sae_i.W_dec.detach().cpu().numpy().astype(np.float32),
        w_dec_txt=sae_t.W_dec.detach().cpu().numpy().astype(np.float32),
        b_dec_img=sae_i.b_dec.detach().cpu().numpy().astype(np.float32),
        b_dec_txt=sae_t.b_dec.detach().cpu().numpy().astype(np.float32),
        w_enc_img=sae_i.encoder.weight.detach().cpu().numpy().astype(np.float32),
        w_enc_txt=sae_t.encoder.weight.detach().cpu().numpy().astype(np.float32),
        b_enc_img=sae_i.encoder.bias.detach().cpu().numpy().astype(np.float32),
        b_enc_txt=sae_t.encoder.bias.detach().cpu().numpy().astype(np.float32),
        phi_S=builder.phi_S.astype(np.float32),
        psi_S=builder.psi_S.astype(np.float32),
        phi_I=(builder.phi_I.astype(np.float32) if builder.phi_I is not None else np.zeros((0, 0), dtype=np.float32)),
        psi_T=(builder.psi_T.astype(np.float32) if builder.psi_T is not None else np.zeros((0, 0), dtype=np.float32)),
        alpha_target=np.array(alpha, dtype=np.float32),
        seed=np.array(seed, dtype=np.int32),
        latent_size_img=np.array(sae_i.latent_size, dtype=np.int32),
        latent_size_txt=np.array(sae_t.latent_size, dtype=np.int32),
        same_model_flag=np.array(int(sae_i is sae_t), dtype=np.int32),
    )


_AUX_FN_MAP = {
    "group_sparse": group_sparse_loss,
    "trace_align": trace_alignment_loss,
    "iso_align": iso_alignment_penalty,
}


def _train_method(
    method: MethodConfig,
    cfg: ExperimentConfig,
    train_ds: SyntheticPairedDataset,
    eval_ds: SyntheticPairedDataset,
    train_img: torch.Tensor,
    train_txt: torch.Tensor,
    latent_size: int,
    seed: int,
    run_dir: Path,
    device: torch.device,
) -> tuple[TopKSAE, TopKSAE, str]:
    """Train one method. Returns ``(sae_i, sae_t, method_id)``."""
    d = cfg.data
    t = cfg.training
    method_id = method.name

    tmp_dir = str(run_dir / "tmp" / method_id)
    training_args = _make_training_args(cfg, tmp_dir, seed)
    callbacks = [SAENormCallback()]

    if method.name == "single_recon":
        _seed_everything(seed)
        model_cfg = TopKSAEConfig(
            hidden_size=d.representation_dim, latent_size=latent_size,
            k=t.k, normalize_decoder=True, weight_tie=t.weight_tie,
        )
        model = TopKSAE(model_cfg).to(device)
        trainer = SingleReconTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, data_collator=default_data_collator,
            callbacks=callbacks,
        )
        trainer.train()
        return model, model, method_id

    elif method.name == "two_recon":
        _seed_everything(seed)
        model_cfg = TwoSidedTopKSAEConfig(
            hidden_size=d.representation_dim, latent_size=latent_size,
            k=t.k, normalize_decoder=True, weight_tie=t.weight_tie,
        )
        model = TwoSidedTopKSAE(model_cfg).to(device)
        trainer = TwoReconTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, data_collator=default_data_collator,
            callbacks=callbacks,
        )
        trainer.train()
        return model.image_sae, model.text_sae, method_id

    elif method.name in _AUX_FN_MAP:
        _seed_everything(seed)
        model_cfg = TopKSAEConfig(
            hidden_size=d.representation_dim, latent_size=latent_size,
            k=t.k, normalize_decoder=True, weight_tie=t.weight_tie,
        )
        model = TopKSAE(model_cfg).to(device)
        aux_fn = _AUX_FN_MAP[method.name]
        method_id = f"{method.name}_w{method.aux_weight}"
        trainer = PairedAuxTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, data_collator=default_data_collator,
            callbacks=callbacks,
            aux_fn=aux_fn, aux_weight=method.aux_weight,
        )
        trainer.train()
        return model, model, method_id

    elif method.name == "ours":
        _seed_everything(seed)
        model_cfg = TwoSidedTopKSAEConfig(
            hidden_size=d.representation_dim, latent_size=latent_size,
            k=t.k, normalize_decoder=True, weight_tie=t.weight_tie,
        )
        model = TwoSidedTopKSAE(model_cfg).to(device)
        method_id = f"ours_l{method.lambda_aux}_m{method.m_S}"
        trainer = OursTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, data_collator=default_data_collator,
            k_align=method.k_align, lambda_aux=method.lambda_aux,
            m_S=method.m_S, aux_norm=method.aux_norm,
            train_img=train_img, train_txt=train_txt, rho=cfg.diagnostic.rho,
        )
        perm_cb = PermutationCallback(trainer)
        trainer.add_callback(perm_cb)
        trainer.add_callback(SAENormCallback())
        trainer.train()
        return model.image_sae, model.text_sae, method_id

    else:
        raise ValueError(f"Unknown method: {method.name}")


# ------------------------------------------------------------------
# Run one (alpha, latent_size, seed)
# ------------------------------------------------------------------


def run_single(
    cfg: ExperimentConfig,
    alpha: float,
    latent_size: int,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    device = _resolve_device(cfg.training.device)
    d = cfg.data

    builder = SyntheticPairedBuilder(
        n_shared=d.n_shared, n_image=d.n_image, n_text=d.n_text,
        representation_dim=d.representation_dim, sparsity=d.sparsity,
        beta=d.beta, obs_noise_std=d.obs_noise_std,
        max_interference=d.max_interference,
        alpha_target=alpha, num_train=d.num_train, num_eval=d.num_eval,
        seed=seed,
    )
    ds = builder.build()
    if getattr(d, "l2_normalize", False):
        for split in ("train", "eval"):
            for key in ("image_representation", "text_representation"):
                arr = ds[split][key]
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                ds[split][key] = (arr / norms).astype(arr.dtype, copy=False)
    train_ds = SyntheticPairedDataset(ds["train"]["image_representation"], ds["train"]["text_representation"])
    eval_ds = SyntheticPairedDataset(ds["eval"]["image_representation"], ds["eval"]["text_representation"])
    train_img = torch.from_numpy(ds["train"]["image_representation"])
    train_txt = torch.from_numpy(ds["train"]["text_representation"])
    eval_img = torch.from_numpy(ds["eval"]["image_representation"])
    eval_txt = torch.from_numpy(ds["eval"]["text_representation"])

    result: dict[str, Any] = {
        "seed": seed,
        "alpha_target": alpha,
        "latent_size": latent_size,
        "alpha_actual_mean": float(builder.mean_shared_cosine_similarity),
        "alpha_actual_std": float(builder.std_shared_cosine_similarity),
    }

    for method in cfg.methods:
        try:
            sae_i, sae_t, method_id = _train_method(
                method, cfg, train_ds, eval_ds,
                train_img, train_txt,
                latent_size, seed, run_dir, device,
            )
        except NotImplementedError as e:
            logger.warning("Skipping %s: %s", method.name, e)
            continue

        metrics = evaluate_method(
            method=method.name,
            sae_i=sae_i, sae_t=sae_t,
            train_img=train_img, train_txt=train_txt,
            eval_img=eval_img, eval_txt=eval_txt,
            phi_S=builder.phi_S, psi_S=builder.psi_S,
            phi_I=builder.phi_I, psi_T=builder.psi_T,
            n_shared=d.n_shared, m_S=method.m_S,
            batch_size=cfg.training.batch_size, device=device,
        )
        result[method_id] = metrics

        if cfg.output.save_decoders:
            _save_params(sae_i, sae_t, builder, alpha, seed, method_id, run_dir)

        if sae_i is not sae_t:
            del sae_i
        del sae_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


# ------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------


def run_experiment(cfg: ExperimentConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{cfg.output.run_tag}" if cfg.output.run_tag else ""
    run_name = f"run_{timestamp}{tag}"
    run_dir = Path(cfg.output.root) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output: %s", run_dir)

    all_results: list[dict[str, Any]] = []
    method_ids: list[str] = []

    for alpha in cfg.sweep.alpha:
        for latent_size in cfg.sweep.latent_size:
            seed_results: list[dict[str, Any]] = []
            for si in range(cfg.sweep.num_seeds):
                seed = cfg.sweep.seed_base + si
                logger.info("alpha=%.2f L=%d seed=%d", alpha, latent_size, seed)
                r = run_single(cfg, alpha, latent_size, seed, run_dir)
                seed_results.append(r)

                for k in r:
                    if k not in ("seed", "alpha_target", "latent_size",
                                 "alpha_actual_mean", "alpha_actual_std"):
                        if k not in method_ids:
                            method_ids.append(k)

            agg = aggregate(seed_results, method_ids)
            entry = {
                "alpha_target": alpha,
                "latent_size": latent_size,
                "aggregate": agg,
                "per_seed": seed_results,
            }
            all_results.append(entry)

    out = {"sweep_results": all_results, "config": _cfg_to_dict(cfg)}
    with open(run_dir / "result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info("Saved result.json to %s", run_dir / "result.json")


def _cfg_to_dict(cfg: ExperimentConfig) -> dict:
    from dataclasses import asdict
    return {
        "data": asdict(cfg.data),
        "training": asdict(cfg.training),
        "methods": [asdict(m) for m in cfg.methods],
        "sweep": asdict(cfg.sweep),
        "output": asdict(cfg.output),
        "diagnostic": asdict(cfg.diagnostic),
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic Theorem 2 experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
