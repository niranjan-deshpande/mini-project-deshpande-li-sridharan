"""Decomposition analyses for the reward-hacking direction (Phase 4 items 1-6, Phase 5 emotion regression)."""
from .confounds import confound_crosstab, confound_response_auroc, stratified_auroc
from .decision import (add_decomposition, clean_z_params, compare_max_detector, max_detector, peak_decomposition,
                       pool_max, project, project_rollouts, summarize_peaks)
from .directions import Reliability, dom, label_shuffled_doms, spearman_brown, split_half_reliability
from .emotions import fit_penalized, selection_frequencies
from .functional import functional_r2, functional_r2_bootstrap, group_moments
from .geometry import (SpanFit, ceiling_r2, cosine, disattenuated_cosine, gram_r2, null_span_r2, span_fit,
                       stack_columns)
from .shapley import shapley_functional, shapley_geo
from .stats import (auroc, aurocs, length_residualize, paired_task_bootstrap, percentile_ci, split_masks,
                    task_half)
