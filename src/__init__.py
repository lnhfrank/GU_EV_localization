"""
src — GDGU EVCS Localization (multi-label graph classification)

Public API:

  Models
    GCN_Graph, GAT_Graph, GIN_Graph, MODEL_CLASSES, AuxWrapper

  Data
    load_evcs_data, build_graphs_route_a,
    augment_route_a,
    stratified_split_multilabel, fit_scaler,
    EVCS_PRESETS

  Training
    train_model_joint,
    evaluate_model, evaluate_aux_acc,
    compute_mia_auc, get_pos_weights,
    kaiming_init, save_checkpoint

  Unlearning
    gdgu_feature_unlearn,
    gif_unlearn, idea_unlearn,
    compute_batch_gradient,
    recalibrate_batchnorm,
    finetune_after_gdgu

  Privacy
    L2_a_integrated_gradients, L2_b_occlusion_delta_auc,
    L2_c_reconstruction, L2_e_attack_type_inference,
    extract_graph_embeddings, derive_det_from_loc,
    measure_unlearn_efficiency

  Experiment
    run_single_trial_route_a
"""

from .models import GCN_Graph, GAT_Graph, GIN_Graph, MODEL_CLASSES, AuxWrapper

from .data import (
    load_evcs_data,
    build_graphs_route_a,
    augment_route_a,
    stratified_split_multilabel,
    fit_scaler,
    EVCS_PRESETS,
)

from .training import (
    train_model_joint,
    evaluate_model,
    evaluate_aux_acc,
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

from .privacy import (
    L2_a_integrated_gradients,
    L2_b_occlusion_delta_auc,
    L2_c_reconstruction,
    L2_e_attack_type_inference,
    extract_graph_embeddings,
    derive_det_from_loc,
    measure_unlearn_efficiency,
)

from .experiment import run_single_trial_route_a

__all__ = [
    # Models
    "GCN_Graph", "GAT_Graph", "GIN_Graph", "MODEL_CLASSES", "AuxWrapper",
    # Data
    "load_evcs_data", "build_graphs_route_a",
    "augment_route_a",
    "stratified_split_multilabel", "fit_scaler", "EVCS_PRESETS",
    # Training
    "train_model_joint",
    "evaluate_model", "evaluate_aux_acc",
    "compute_mia_auc", "get_pos_weights",
    "kaiming_init", "save_checkpoint",
    # Unlearning
    "gdgu_feature_unlearn", "gif_unlearn", "idea_unlearn",
    "compute_batch_gradient", "recalibrate_batchnorm", "finetune_after_gdgu",
    # Privacy
    "L2_a_integrated_gradients", "L2_b_occlusion_delta_auc",
    "L2_c_reconstruction", "L2_e_attack_type_inference",
    "extract_graph_embeddings", "derive_det_from_loc",
    "measure_unlearn_efficiency",
    # Experiment
    "run_single_trial_route_a",
]
