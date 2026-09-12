"""Smoke mode: the slice knobs, the streaming COCO read, and the refusal.

Smoke mode exists so that the whole post-rebuttal run can be exercised on a
small slice of the real Hugging Face data, with the real encoder and the real
model sizes, before the full run is started. Three things have to hold and are
checked here.

  The knobs reach the extractors. `cache.max_samples` bounds the training
  extraction, `eval.max_samples` bounds the two evaluation extractions, and
  both pipelines pass them through.

  A slice does not cost a full download. Asking COCO for a slice switches the
  source to streaming parquet reads, which fetch only the row groups they
  consume. A full run must still read it the way it always has.

  A sliced cache cannot be mistaken for a full one. Every extractor records its
  slice in meta.json, and a pipeline whose config asks for something else
  refuses by name rather than training on what it found. The refusal covers
  exactly the caches a run reads: a cache no configured evaluation opens is not
  checked, and the rebuttal stage checks its own caches even when it runs alone.

Nothing is downloaded and no encoder is loaded: the source dataset and the
encoder are both stubs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data import extract
from src.data.cache_io import cache_slice_mismatch, require_cache_slice
from src.utils.config import (
    CacheConfig,
    Config,
    EvalConfig,
    MethodConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    load_config,
)
from tests.conftest import make_coco_cache, make_imagenet_cache

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

DIM = 4


# --------------------------------------------------------------------------- #
# the config keys
# --------------------------------------------------------------------------- #
def test_the_slice_knobs_default_to_taking_everything() -> None:
    """With nothing set, every knob means "the whole corpus"."""
    assert CacheConfig(cache_dir="x").max_samples is None
    assert EvalConfig().max_samples is None
    assert EvalConfig().recon_imagenet is True
    assert EvalConfig().coco_cache_dir == "cache/{key}_coco"
    assert EvalConfig().imagenet_cache_dir == "cache/{key}_imagenet"


def test_the_full_run_config_sets_none_of_them() -> None:
    """The shipped full run must keep taking the whole corpus."""
    full = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    assert full.rebuttal.coco80_min_count is None
    assert full.figure2.cache.max_samples is None
    assert full.table1.cache.max_samples is None
    assert full.table1.eval.max_samples is None
    assert full.table1.eval.recon_imagenet is True
    assert full.table1.eval.coco_cache_dir == "cache/{key}_coco"
    assert full.table1.eval.imagenet_cache_dir == "cache/{key}_imagenet"


def test_the_smoke_config_carries_the_slices_it_claims() -> None:
    """Every value the smoke run depends on, read back off the YAML."""
    cfg = load_config(CONFIGS / "post_rebuttal" / "smoke.yaml")
    assert cfg.kind == "post_rebuttal"
    assert cfg.output.root == "outputs/post_rebuttal_smoke"
    assert cfg.rebuttal.tau == 0.4
    assert cfg.rebuttal.null_seed == 7
    assert cfg.rebuttal.n_boot == 200
    assert cfg.rebuttal.coco_seed_b == 1
    assert cfg.rebuttal.coco80_min_count == 20

    coco = cfg.figure2
    assert coco.kind == "multi_density"
    assert [m.key for m in coco.models] == ["clip_b32"]
    assert coco.cache.cache_dir == "cache/smoke/{key}_coco"
    assert coco.cache.max_samples == 2000
    assert coco.training.num_epochs == 2
    assert coco.output.root == "outputs/post_rebuttal_smoke/coco_clip_b32"

    cc3m = cfg.table1
    assert cc3m.kind == "cc3m_downstream"
    assert cc3m.cache.cache_dir == "cache/smoke/clip_b32_cc3m"
    assert cc3m.cache.max_samples == 6000
    assert cc3m.training.num_epochs == 1
    assert cc3m.training.resolved_seeds() == [0, 1]
    assert cc3m.eval.max_samples == 2000
    assert cc3m.eval.coco_cache_dir == "cache/smoke/clip_b32_coco"
    assert cc3m.eval.imagenet_cache_dir == "cache/smoke/clip_b32_imagenet"
    assert cc3m.output.root == "outputs/post_rebuttal_smoke/cc3m_clip_b32"


def test_the_smoke_run_keeps_the_model_size_of_the_full_run() -> None:
    """Only the amount of data and the training length may differ.

    The point of the smoke run is to prove the code paths at the size they will
    run at, so the number of active latents per input and the total latent
    budget have to match the full run exactly.
    """
    full = load_config(CONFIGS / "post_rebuttal" / "clip_b32.yaml")
    smoke = load_config(CONFIGS / "post_rebuttal" / "smoke.yaml")
    for part in ("figure2", "table1"):
        a = getattr(full, part).training
        b = getattr(smoke, part).training
        assert (b.k, b.latent_size) == (a.k, a.latent_size), \
            f"the smoke {part} config changed the model size"
    assert [m.key for m in smoke.figure2.models] == \
        [m.key for m in full.figure2.models]
    assert smoke.table1.model.key == full.table1.model.key


def test_the_smoke_caches_are_kept_apart_from_the_full_run() -> None:
    """Nothing the smoke run writes may land where the full run reads."""
    smoke = load_config(CONFIGS / "post_rebuttal" / "smoke.yaml")
    written = [
        smoke.output.root,
        smoke.figure2.output.root,
        smoke.table1.output.root,
        smoke.figure2.cache.cache_dir,
        smoke.table1.cache.cache_dir,
        smoke.table1.eval.coco_cache_dir,
        smoke.table1.eval.imagenet_cache_dir,
    ]
    for path in written:
        assert path.startswith("outputs/post_rebuttal_smoke") or \
            path.startswith("cache/smoke/"), f"{path} is not a smoke-only path"


# --------------------------------------------------------------------------- #
# COCO: streaming only when a slice is asked for
# --------------------------------------------------------------------------- #
class _StubEncoder:
    """Maps an image to its red channel and a caption to its length."""

    dim = DIM
    device = torch.device("cpu")

    def encode_image(self, images):
        vals = [float(im.getpixel((0, 0))[0]) for im in images]
        return torch.tensor([[v] * DIM for v in vals], dtype=torch.float32)

    def encode_text(self, texts):
        return torch.tensor([[float(len(t))] * DIM for t in texts], dtype=torch.float32)


@pytest.fixture
def coco_source(monkeypatch):
    """A stub COCO dataset and encoder. Returns the list of load_dataset calls.

    Each recorded call is the keyword dict `extract_coco` handed to
    `datasets.load_dataset`, so a test can read back whether streaming was
    requested for that split.
    """
    import datasets

    calls: list[dict] = []

    def fake_load_dataset(path, *, split, **kwargs):
        calls.append({"path": path, "split": split, **kwargs})
        base = {"train": 0, "validation": 100, "test": 200}[split]
        return [
            {"image_id": base + i,
             "image": Image.new("RGB", (2, 2), color=(base + i, 0, 0)),
             "captions": [f"caption {c} of {base + i}" for c in range(2)]}
            for i in range(5)
        ]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(extract, "load_encoder", lambda *a, **k: _StubEncoder())
    return calls


def _model_cfg() -> ModelConfig:
    return ModelConfig(key="stub", backend="transformers", hidden_size=DIM)


def test_a_sliced_coco_read_is_streamed_and_records_its_slice(
        tmp_path: Path, coco_source) -> None:
    """A slice reads the parquet rows it consumes and nothing else."""
    cache_dir = tmp_path / "coco"
    extract.extract_coco(model_cfg=_model_cfg(), cache_dir=cache_dir,
                         batch_size=2, device="cpu", max_groups_per_split=2)

    assert [c["split"] for c in coco_source] == ["train", "validation", "test"]
    assert all(c.get("streaming") is True for c in coco_source), \
        "a sliced COCO read must not download the whole parquet set"

    splits = json.loads((cache_dir / "splits.json").read_text())
    # Two photographs per split, two captions each, so four pairs per split.
    assert {k: len(v) for k, v in splits.items()} == {"train": 4, "val": 4, "test": 4}
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["max_samples"] == 2


def test_a_full_coco_read_is_not_streamed_and_records_no_slice(
        tmp_path: Path, coco_source) -> None:
    """The full run must keep reading COCO exactly as it always has."""
    cache_dir = tmp_path / "coco"
    extract.extract_coco(model_cfg=_model_cfg(), cache_dir=cache_dir,
                         batch_size=2, device="cpu")

    assert all("streaming" not in c for c in coco_source), \
        "the full COCO read asked for streaming"
    splits = json.loads((cache_dir / "splits.json").read_text())
    assert {k: len(v) for k, v in splits.items()} == \
        {"train": 10, "val": 10, "test": 10}
    meta = json.loads((cache_dir / "meta.json").read_text())
    assert meta["max_samples"] is None


def test_a_sliced_coco_read_resumes_where_it_stopped(
        tmp_path: Path, coco_source, monkeypatch) -> None:
    """Groups are counted the same way in both modes, so a resume still lands.

    The slice is applied after the groups already on disk are skipped, so a
    pass that was interrupted at one photograph and is asked for three takes
    two more, not three.
    """
    import functools

    monkeypatch.setattr(
        extract, "_ChunkWriter",
        functools.partial(extract._ChunkWriter, chunk_size=1),
    )
    cache_dir = tmp_path / "coco"
    extract.extract_coco(model_cfg=_model_cfg(), cache_dir=cache_dir,
                         batch_size=1, device="cpu", max_groups_per_split=1)
    first = json.loads((cache_dir / "splits.json").read_text())
    assert {k: len(v) for k, v in first.items()} == {"train": 2, "val": 2, "test": 2}

    # The finished cache is idempotent, so the second pass is made against the
    # chunk files of a split that was never assembled.
    for name in ("image_embeddings.npy", "text_embeddings.npy", "keys.json",
                 "splits.json", "meta.json", "captions.json"):
        (cache_dir / name).unlink()
    extract.extract_coco(model_cfg=_model_cfg(), cache_dir=cache_dir,
                         batch_size=1, device="cpu", max_groups_per_split=3)

    keys = json.loads((cache_dir / "keys.json").read_text())
    captions = json.loads((cache_dir / "captions.json").read_text())
    assert [k for k in keys if k.startswith(("0_", "1_", "2_"))] == \
        ["0_0", "0_1", "1_0", "1_1", "2_0", "2_1"]
    assert sorted(captions) == sorted(keys), "the resumed slice lost captions"


# --------------------------------------------------------------------------- #
# a sliced cache is recognisable
# --------------------------------------------------------------------------- #
def _write_meta(cache_dir: Path, max_samples: int | None, *, record: bool = True) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model_key": "stub", "dim": DIM, "dataset": "coco"}
    if record:
        meta["max_samples"] = max_samples
    (cache_dir / "meta.json").write_text(json.dumps(meta))


def test_a_matching_slice_is_accepted(tmp_path: Path) -> None:
    _write_meta(tmp_path, 2000)
    assert cache_slice_mismatch(tmp_path, 2000) is None
    _write_meta(tmp_path, None)
    assert cache_slice_mismatch(tmp_path, None) is None


def test_a_cache_written_before_the_field_existed_reads_as_a_full_one(
        tmp_path: Path) -> None:
    _write_meta(tmp_path, None, record=False)
    assert cache_slice_mismatch(tmp_path, None) is None
    assert cache_slice_mismatch(tmp_path, 2000) is not None


def test_a_missing_meta_file_is_not_an_error(tmp_path: Path) -> None:
    """Nothing has been extracted yet, so there is nothing to disagree with."""
    assert cache_slice_mismatch(tmp_path / "nothing", 2000) is None


def test_a_sliced_cache_is_refused_by_a_full_run(tmp_path: Path) -> None:
    _write_meta(tmp_path, 2000)
    why = cache_slice_mismatch(tmp_path, None)
    assert why is not None
    assert str(tmp_path) in why
    assert "only the first 2,000 source records" in why
    assert "whole corpus" in why
    assert "Delete" in why
    with pytest.raises(ValueError, match="Delete"):
        require_cache_slice(tmp_path, None)


def test_a_full_cache_is_refused_by_a_sliced_run(tmp_path: Path) -> None:
    _write_meta(tmp_path, None)
    why = cache_slice_mismatch(tmp_path, 500)
    assert why is not None and "the first 500 source records" in why
    with pytest.raises(ValueError):
        require_cache_slice(tmp_path, 500)


def test_two_different_slices_are_refused(tmp_path: Path) -> None:
    _write_meta(tmp_path, 6000)
    assert cache_slice_mismatch(tmp_path, 2000) is not None


# --------------------------------------------------------------------------- #
# the pipelines: the knobs are threaded, and a wrong cache is refused
# --------------------------------------------------------------------------- #
def _smoke_config():
    return load_config(CONFIGS / "post_rebuttal" / "smoke.yaml")


def test_the_figure2_pipeline_passes_its_slice_to_the_extractor(
        tmp_path: Path, monkeypatch) -> None:
    """The smoke config reaches `extract_cache` with the slice it declares."""
    from src.pipelines import multi_density

    seen: list[dict] = []
    monkeypatch.setattr(multi_density, "extract_cache",
                        lambda **kw: seen.append(kw))
    monkeypatch.chdir(tmp_path)
    multi_density.run(_smoke_config().figure2, stage="extract")

    assert len(seen) == 1
    assert seen[0]["max_samples"] == 2000
    assert seen[0]["cache_cfg"].cache_dir == "cache/smoke/clip_b32_coco"
    assert seen[0]["cache_cfg"].max_samples == 2000


def test_the_table1_pipeline_passes_its_slice_to_the_extractor(
        tmp_path: Path, monkeypatch) -> None:
    from src.pipelines import cc3m_downstream

    seen: list[dict] = []
    monkeypatch.setattr(cc3m_downstream, "extract_cache",
                        lambda **kw: seen.append(kw))
    monkeypatch.chdir(tmp_path)
    cc3m_downstream.run(_smoke_config().table1, stage="extract")

    assert len(seen) == 1
    assert seen[0]["max_samples"] == 6000
    assert seen[0]["cache_cfg"].cache_dir == "cache/smoke/clip_b32_cc3m"


def test_the_figure2_pipeline_refuses_a_cache_from_another_slice(
        tmp_path: Path, monkeypatch) -> None:
    """A full cache already on disk must not be trained on by a sliced run."""
    from src.pipelines import multi_density

    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache" / "smoke" / "clip_b32_coco"
    make_coco_cache(cache_dir, n_images=6, caps_per_image=2, dim=DIM)
    monkeypatch.setattr(multi_density, "extract_cache",
                        lambda **kw: pytest.fail("extraction must not start"))
    with pytest.raises(ValueError, match="clip_b32_coco"):
        multi_density.run(_smoke_config().figure2, stage="extract")


def test_the_table1_pipeline_refuses_a_cache_from_another_slice(
        tmp_path: Path, monkeypatch) -> None:
    from src.pipelines import cc3m_downstream

    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache" / "smoke" / "clip_b32_cc3m"
    make_coco_cache(cache_dir, n_images=6, caps_per_image=1, dim=DIM)
    monkeypatch.setattr(cc3m_downstream, "extract_cache",
                        lambda **kw: pytest.fail("extraction must not start"))
    with pytest.raises(ValueError, match="clip_b32_cc3m"):
        cc3m_downstream.run(_smoke_config().table1, stage="extract")


def test_the_smoke_config_resolves_to_smoke_only_settings() -> None:
    """The rebuttal stage reads only the smoke run's own caches and outputs."""
    from src.rebuttal.common import settings_from_config

    settings = settings_from_config(_smoke_config())
    assert sorted(settings) == ["cc3m_k32", "coco_k8"]
    for setting in settings.values():
        for path in setting.as_dict().values():
            if isinstance(path, str) and ("cache/" in path or "outputs/" in path):
                assert "smoke" in path, f"{path} points outside the smoke run"
    assert settings["coco_k8"].k == 8
    assert settings["cc3m_k32"].k == 32
    assert settings["coco_k8"].latent_size == 8192
    # The second CC3M seed is what the same-modality comparison needs, so the
    # two seeds of the smoke config have to land on two different checkpoints.
    assert settings["cc3m_k32"].ckpt_a != settings["cc3m_k32"].ckpt_b


def test_the_evaluation_caches_are_named_by_the_config() -> None:
    from src.pipelines.cc3m_downstream import _coco_cache_dir, _imagenet_cache_dir

    default = EvalConfig()
    assert _coco_cache_dir(default, "clip_b32") == Path("cache/clip_b32_coco")
    assert _imagenet_cache_dir(default, "clip_b32") == Path("cache/clip_b32_imagenet")

    smoke = _smoke_config().table1.eval
    assert _coco_cache_dir(smoke, "clip_b32") == Path("cache/smoke/clip_b32_coco")
    assert _imagenet_cache_dir(smoke, "clip_b32") == \
        Path("cache/smoke/clip_b32_imagenet")


def test_imagenet_is_extracted_only_when_an_evaluation_reads_it() -> None:
    """Zero-shot and the ImageNet half of reconstruction are the only two."""
    from src.pipelines.cc3m_downstream import _wants_imagenet

    assert _wants_imagenet(EvalConfig())
    assert _wants_imagenet(EvalConfig(zeroshot=True, recon=False))
    assert _wants_imagenet(EvalConfig(zeroshot=False, recon=True,
                                      recon_imagenet=True))
    assert not _wants_imagenet(EvalConfig(zeroshot=False, recon=True,
                                          recon_imagenet=False))
    assert not _wants_imagenet(EvalConfig(zeroshot=False, recon=False))


def test_the_coco80_threshold_reaches_the_analyses_only_when_it_is_set(
        tmp_path: Path, monkeypatch) -> None:
    """Left unset, each COCO-80 analysis keeps its own default of 50.

    The smoke run lowers it because its COCO test split holds 2,000
    photographs rather than 5,000, and on that slice no object category reaches
    50 positives in each half, which makes both analyses raise.
    """
    import sys
    import types

    from src.pipelines.post_rebuttal import run as post_rebuttal_run
    from src.rebuttal import registry
    from src.rebuttal.registry import Analysis
    from tests.test_post_rebuttal_smoke import _place_checkpoints, _tiny_config

    seen: list[dict] = []

    def works(setting, *, out_dir, device, **knobs) -> dict:
        seen.append(dict(knobs))
        (Path(out_dir) / "works.json").write_text("{}")
        (Path(out_dir) / "works.md").write_text("# Works\n\nMeasured 0 rows.\n")
        return {}

    module = types.ModuleType("tests_min_count")
    module.run = works
    monkeypatch.setitem(sys.modules, "tests_min_count", module)
    monkeypatch.setattr(registry, "ANALYSES",
                        [Analysis("works", "tests_min_count", ("coco_k8",))])

    monkeypatch.chdir(tmp_path)
    make_coco_cache(tmp_path / "cache" / "clip_b32_coco", n_images=8,
                    caps_per_image=2, dim=16)
    cfg = _tiny_config(tmp_path)
    cfg.rebuttal.settings = ["coco_k8"]      # only the COCO cache is written here
    _place_checkpoints(cfg)

    post_rebuttal_run(cfg, stage="rebuttal")
    assert "min_count" not in seen[-1]

    (tmp_path / "out" / "rebuttal" / "coco_k8" / "works.json").unlink()
    cfg.rebuttal.coco80_min_count = 20
    post_rebuttal_run(cfg, stage="rebuttal")
    assert seen[-1]["min_count"] == 20


# --------------------------------------------------------------------------- #
# the refusal covers exactly the caches a run reads
# --------------------------------------------------------------------------- #
def _record_slice(cache_dir: Path, max_samples: int | None) -> None:
    """Set the slice a finished fixture cache claims to hold."""
    meta = json.loads((cache_dir / "meta.json").read_text())
    meta["max_samples"] = max_samples
    (cache_dir / "meta.json").write_text(json.dumps(meta))


def _eval_only_config(tmp_path: Path, **eval_kw) -> Config:
    """A cc3m_downstream config whose evaluation caches are the smoke ones.

    No checkpoint is placed, so every method is skipped and the stage runs only
    the cache checks and the two on-demand extractions.
    """
    return Config(
        kind="cc3m_downstream",
        model=ModelConfig(key="clip_b32", backend="transformers", hidden_size=DIM),
        cache=CacheConfig(cache_dir="cache/smoke/clip_b32_cc3m", dataset="cc3m",
                          split="train"),
        training=TrainingConfig(seed=0, seeds=[0], lr=1e-3, num_epochs=1,
                                batch_size=8, k=2, latent_size=8, device="cpu"),
        methods=[MethodConfig(name="separated")],
        eval=EvalConfig(max_samples=2000, coco_cache_dir="cache/smoke/{key}_coco",
                        **eval_kw),
        output=OutputConfig(root=str(tmp_path / "out")),
    )


def test_an_imagenet_cache_that_no_evaluation_reads_is_not_checked(
        tmp_path: Path, monkeypatch) -> None:
    """A run with both ImageNet evaluations off ignores the ImageNet cache.

    This is the combination a collaborator without a Hugging Face token is told
    to use. Refusing it over the slice of a full ImageNet cache the run never
    opens would leave two wrong ways out: delete a multi-gigabyte cache that is
    not being used, or widen the slice of the caches that are.
    """
    from src.pipelines import cc3m_downstream

    monkeypatch.chdir(tmp_path)
    coco_dir = tmp_path / "cache" / "smoke" / "clip_b32_coco"
    make_coco_cache(coco_dir, n_images=4, caps_per_image=2, dim=DIM)
    _record_slice(coco_dir, 2000)
    make_imagenet_cache(tmp_path / "cache" / "clip_b32_imagenet", n_images=4,
                        n_classes=2, n_templates=2, dim=DIM)   # the whole corpus
    monkeypatch.setattr(extract, "extract_imagenet",
                        lambda **kw: pytest.fail("ImageNet must not be extracted"))

    cfg = _eval_only_config(tmp_path, recon=True, retrieval=True, zeroshot=False,
                            recon_imagenet=False)
    cc3m_downstream._eval_stage(cfg, tmp_path / "out" / "seed0")


def test_an_imagenet_cache_an_evaluation_reads_is_still_refused(
        tmp_path: Path, monkeypatch) -> None:
    """With zero-shot on, the same mismatched ImageNet cache stops the run."""
    from src.pipelines import cc3m_downstream

    monkeypatch.chdir(tmp_path)
    coco_dir = tmp_path / "cache" / "smoke" / "clip_b32_coco"
    make_coco_cache(coco_dir, n_images=4, caps_per_image=2, dim=DIM)
    _record_slice(coco_dir, 2000)
    make_imagenet_cache(tmp_path / "cache" / "clip_b32_imagenet", n_images=4,
                        n_classes=2, n_templates=2, dim=DIM)

    cfg = _eval_only_config(tmp_path, recon=False, retrieval=True, zeroshot=True)
    with pytest.raises(ValueError, match="clip_b32_imagenet"):
        cc3m_downstream._eval_stage(cfg, tmp_path / "out" / "seed0")


def test_a_coco_cache_no_evaluation_reads_is_not_checked(
        tmp_path: Path, monkeypatch) -> None:
    """The COCO cache is checked under the same rule as the ImageNet one."""
    from src.pipelines import cc3m_downstream

    monkeypatch.chdir(tmp_path)
    coco_dir = tmp_path / "cache" / "smoke" / "clip_b32_coco"
    make_coco_cache(coco_dir, n_images=4, caps_per_image=2, dim=DIM)   # full
    monkeypatch.setattr(extract, "extract_coco",
                        lambda **kw: pytest.fail("COCO must not be extracted"))

    cfg = _eval_only_config(tmp_path, recon=False, retrieval=False, zeroshot=False,
                            recon_imagenet=False)
    cc3m_downstream._eval_stage(cfg, tmp_path / "out" / "seed0")


def test_a_slice_smaller_than_an_interrupted_pass_is_refused(
        tmp_path: Path, coco_source, monkeypatch) -> None:
    """A resume cannot assemble more source records than meta.json will record.

    A pass killed before assembly leaves its chunk files behind. A later pass
    asking for a smaller slice skips every one of those records and takes
    nothing further, so it would assemble the larger cache and label it with the
    smaller number, which every later check reads. It refuses instead.
    """
    cache_dir = tmp_path / "coco"
    model = _model_cfg()

    def killed_before_assembly(*args, **kwargs):
        raise RuntimeError("killed before assembly")

    monkeypatch.setattr(extract, "_assemble", killed_before_assembly)
    with pytest.raises(RuntimeError, match="killed before assembly"):
        extract.extract_coco(model_cfg=model, cache_dir=cache_dir, batch_size=2,
                             device="cpu")                      # all 5 per split
    monkeypatch.undo()
    monkeypatch.setattr(extract, "load_encoder", lambda *a, **k: _StubEncoder())

    with pytest.raises(ValueError, match="already holds 5 source records"):
        extract.extract_coco(model_cfg=model, cache_dir=cache_dir, batch_size=2,
                             device="cpu", max_groups_per_split=2)
    assert not (cache_dir / "meta.json").exists(), "a cache was written anyway"

    # The message names the two ways out. Asking for at least what is cached
    # finishes the pass, and the cache then holds no more than it records.
    extract.extract_coco(model_cfg=model, cache_dir=cache_dir, batch_size=2,
                         device="cpu", max_groups_per_split=5)
    splits = json.loads((cache_dir / "splits.json").read_text())
    assert {k: len(v) for k, v in splits.items()} == \
        {"train": 10, "val": 10, "test": 10}
    assert json.loads((cache_dir / "meta.json").read_text())["max_samples"] == 5


def test_the_rebuttal_stage_refuses_a_cache_from_another_slice(
        tmp_path: Path, monkeypatch) -> None:
    """`--stage rebuttal` alone runs neither pipeline, so it checks for itself.

    A sliced COCO cache left at the full run's path, by someone who asked for a
    slice for a quick check and then took the knob back out, would otherwise be
    trained on by model B and reported as the full result.
    """
    from src.pipelines.post_rebuttal import run as post_rebuttal_run
    from src.rebuttal import registry
    from tests.test_post_rebuttal_smoke import _place_checkpoints, _tiny_config

    monkeypatch.chdir(tmp_path)
    # The analyses themselves are not what this test measures, and they read
    # the Figure 2 panel, which only that pipeline writes.
    monkeypatch.setattr(registry, "ANALYSES", [])
    coco_dir = tmp_path / "cache" / "clip_b32_coco"
    make_coco_cache(coco_dir, n_images=8, caps_per_image=2, dim=16)
    _record_slice(coco_dir, 2000)

    cfg = _tiny_config(tmp_path)             # asks for the whole corpus
    cfg.rebuttal.settings = ["coco_k8"]
    _place_checkpoints(cfg)
    with pytest.raises(ValueError, match="clip_b32_coco"):
        post_rebuttal_run(cfg, stage="rebuttal")
    assert not (tmp_path / "out" / "rebuttal" / "coco_k8" / "seed1").exists(), \
        "model B was trained on a cache from another slice"

    # The same stage runs once the config asks for what the cache holds.
    cfg.figure2.cache.max_samples = 2000
    post_rebuttal_run(cfg, stage="rebuttal")
    assert (tmp_path / "out" / "rebuttal" / "coco_k8" / "panels").exists()
