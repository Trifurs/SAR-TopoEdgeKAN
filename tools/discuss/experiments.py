"""Stable, opt-in experiment groups and their common-protocol checks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from utils.config import load_config
from utils.experiment_catalog import experiment_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs" / "discuss"
GROUPS = {
    "input": (
        "full", "no_angles", "dlsim_attention_unet_matched", "dlsim_linknet_matched",
        "resnet18_depth_regression_matched", "unet_depth_regression_matched",
        "unetplusplus_depth_regression_matched",
    ),
    "graph": ("full", "no_graph", "fixed_edges", "linear_edges", "mlp_edges"),
    "prior": (
        "full", "prior_only", "no_prior", "terrain_only",
        "unet_depth_regression_matched", "unet_shared_head", "unet_prior",
    ),
}


def config_paths(groups=("input", "graph", "prior"), *, trainable_only=False):
    selected = tuple(GROUPS) if "all" in groups else tuple(groups)
    unknown = set(selected).difference(GROUPS)
    if unknown:
        raise ValueError(f"Unknown discussion groups: {sorted(unknown)}")
    names = dict.fromkeys(name for group in selected for name in GROUPS[group])
    paths = tuple((CONFIG_ROOT / f"{name}.xml").resolve() for name in names)
    if trainable_only:
        paths = tuple(path for path in paths if load_config(path)["model"]["discuss"]["trainable"])
    return paths


def validate_protocol(paths):
    """Fail before a campaign if a non-intervention training/data setting differs."""

    reference = load_config(CONFIG_ROOT / "full.xml")
    observed = set()
    for path in paths:
        config = load_config(path)
        identifier = experiment_id(config)
        if identifier in observed:
            raise ValueError(f"Duplicate discussion experiment ID: {identifier}")
        observed.add(identifier)
        for section in ("loss", "optimizer", "scheduler", "training", "supervision"):
            if config[section] != reference[section]:
                raise ValueError(f"{path.name} changes shared section {section}")
        if config["seed"] != reference["seed"]:
            raise ValueError(f"{path.name} changes the paired seed")
        for key in ("device", "deterministic"):
            if config[key] != reference[key]:
                raise ValueError(f"{path.name} changes shared setting {key}")
        if config["runtime"]["random_seed_validation"] != reference["runtime"]["random_seed_validation"]:
            raise ValueError(f"{path.name} changes the paired seed schedule")
        for key in ("trainable_scope", "max_train_batches", "max_val_batches"):
            if config["runtime"]["train"][key] != reference["runtime"]["train"][key]:
                raise ValueError(f"{path.name} changes runtime.train.{key}")
        dataset = deepcopy(config["dataset"])
        if path.stem == "no_angles":
            if dataset["model_bands"]["s1_conditioning"]:
                raise ValueError("The no-angles control still requests incidence angles")
            dataset["model_bands"]["s1_conditioning"] = reference["dataset"]["model_bands"]["s1_conditioning"]
        if dataset != reference["dataset"]:
            raise ValueError(f"{path.name} changes the shared dataset contract or processing")
        if config["model"]["flood_support"] != "valid_depth_mask":
            raise ValueError(f"{path.name} changes the known flood range")
        if config["runtime"]["evaluation"]["split"] != "test":
            raise ValueError(f"{path.name} does not report on the test split")
        if config["runtime"]["train"]["init_checkpoint"] is not None:
            raise ValueError(f"{path.name} does not train from scratch")
    return reference
