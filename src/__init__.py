"""
src — GDGU EVCS Localization (multi-label graph classification)

Public API:

  Models
    GCN_Graph, GAT_Graph, GIN_Graph, MODEL_CLASSES, AuxWrapper

  Data
    load_evcs_data, build_graphs,
    augment_with_power,
    stratified_split_multilabel, fit_scaler,
    EVCS_PRESETS

  Training
    train_model_joint,
    evaluate_model,
    compute_mia_auc, get_pos_weights,
    kaiming_init, save_checkpoint

  Unlearning
    gdgu_feature_unlearn,
    gif_unlearn, idea_unlearn,
    compute_batch_gradient,
    recalibrate_batchnorm,
    finetune_after_gdgu

  Experiment
    run_single_trial
"""

from .models import GCN_Graph, GAT_Graph, GIN_Graph, MODEL_CLASSES, AuxWrapper

from .data import (
    load_evcs_data,
    build_graphs,
    augment_with_power,
    stratified_split_multilabel,
    fit_scaler,
    EVCS_PRESETS,
)

from .training import (
    train_model_joint,
    evaluate_model,
    compute_mia_auc,
    get_pos_weights,
    kaiming_init,
    save_checkpoint,
)

from .unlearning import (
    gdgu_feature_unlearn,
    gif_unlearn,
    idea_unlearn,
    compute_batch_gradient,
    recalibrate_batchnorm,
    finetune_after_gdgu,
)

from .experiment import run_single_trial

__all__ = [
    # Models
    "GCN_Graph", "GAT_Graph", "GIN_Graph", "MODEL_CLASSES", "AuxWrapper",
    # Data
    "load_evcs_data", "build_graphs",
    "augment_with_power",
    "stratified_split_multilabel", "fit_scaler", "EVCS_PRESETS",
    # Training
    "train_model_joint",
    "evaluate_model",
    "compute_mia_auc", "get_pos_weights",
    "kaiming_init", "save_checkpoint",
    # Unlearning
    "gdgu_feature_unlearn", "gif_unlearn", "idea_unlearn",
    "compute_batch_gradient", "recalibrate_batchnorm", "finetune_after_gdgu",
    # Experiment
    "run_single_trial",
]
