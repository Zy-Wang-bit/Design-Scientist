"""External antibody held-out replay benchmark.

This module is deliberately separate from the internal synthetic pH-switch
oracle. It evaluates whether a mechanism can rank held-out, experimentally
measured antibody variants from public antibody fitness landscapes. The first
adapter targets FLAb-style CSV files: ``heavy`` and optional ``light`` sequence
columns plus a numeric ``fitness`` column where higher is better.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from design_scientist.algorithms import pcig


DEFAULT_RUN_ID = "external_antibody_replay"
DEFAULT_PH_SWITCH_RUN_ID = "external_ph_switch_replay"
DEFAULT_SEEDS = range(5)
DEFAULT_BUDGET = 5
DEFAULT_PH_SWITCH_BUDGET = 2
PH_SWITCH_SOURCE_IPI = (
    "Improved therapeutic index of an acidic pH-selective antibody; "
    "PMC8865267 Table 1; https://pmc.ncbi.nlm.nih.gov/articles/PMC8865267/"
)
PH_SWITCH_SOURCE_CEA = (
    "Zou et al. 2022 anti-CEA pH-dependent mutants; PMC8686790 Table 3; "
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8686790/"
)
PH_SWITCH_SOURCE_ALXN1210 = (
    "ALXN1210 C5 pH-dependent binding variants; PLOS ONE 2018 Table 1; "
    "https://doi.org/10.1371/journal.pone.0195909"
)
PH_SWITCH_SOURCE_HER2_BH1 = (
    "Structure-based engineering of pH-dependent antibody binding for selective targeting of solid-tumor "
    "microenvironment; PMC6927761 Table 2; https://pmc.ncbi.nlm.nih.gov/articles/PMC6927761/"
)
PH_SWITCH_SOURCE_ADALIMUMAB_TNF = (
    "Schroeter et al. 2015 adalimumab/rhTNF pH-switch variants; mAbs 2015 Table 1 and structural "
    "discussion; PubMed 25523975; https://pubmed.ncbi.nlm.nih.gov/25523975/"
)
DEFAULT_FLAB_DATASETS: tuple[dict[str, Any], ...] = (
    {
        "dataset_id": "flab_hie2023_s309",
        "source": "FLAb",
        "category": "binding",
        "url": "https://raw.githubusercontent.com/Graylab/FLAb/main/data/binding/hie2023efficient_CoV2_S309_Kd.csv",
        "split_column": "Round",
        "train_values": ["alpha"],
        "test_values": ["beta"],
        "reference": "Hie et al. 2023 efficient evolution; FLAb binding dataset.",
    },
    {
        "dataset_id": "flab_hie2023_medi_h4",
        "source": "FLAb",
        "category": "binding",
        "url": "https://raw.githubusercontent.com/Graylab/FLAb/main/data/binding/hie2023efficient_MEDI_H4Hubei_Kd.csv",
        "split_column": "Round",
        "train_values": ["alpha"],
        "test_values": ["beta"],
        "reference": "Hie et al. 2023 efficient evolution; FLAb binding dataset.",
    },
    {
        "dataset_id": "flab_hutchinson2023_singlekd_fab",
        "source": "FLAb",
        "category": "binding",
        "url": "https://raw.githubusercontent.com/Graylab/FLAb/main/data/binding/hutchinson2023enhancement_singlekd_fab.csv",
        "reference": "Hutchinson et al. 2023 enhancement; FLAb binding dataset.",
    },
)

def _public_ph_switch_row(
    *,
    dataset_id: str,
    source: str,
    variant_id: str,
    mutation_signature: str,
    histidine_count: int,
    acidic_count: int,
    mutation_count: int,
    ratio: float,
    direction: str,
    censored_lower_bound: bool = False,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source": source,
        "variant_id": variant_id,
        "mutation_signature": mutation_signature,
        "histidine_count": histidine_count,
        "acidic_count": acidic_count,
        "mutation_count": mutation_count,
        "ph74_over_ph60_kd_ratio": ratio,
        "selectivity_direction": direction,
        "censored_lower_bound": censored_lower_bound,
    }


DEFAULT_PUBLIC_PH_SWITCH_ROWS: tuple[dict[str, Any], ...] = (
    # Acidic pH-selective CTLA-4 antibody variants from a public table.
    # The reported ratio is neutral/acidity selectivity: larger is stronger
    # neutral-pH loss relative to acidic-pH binding.
    _public_ph_switch_row(
        dataset_id="ipi_acidic_ph_selective_2022",
        source=PH_SWITCH_SOURCE_IPI,
        variant_id="Ipi",
        mutation_signature="WT",
        histidine_count=0,
        acidic_count=0,
        mutation_count=0,
        ratio=1.1,
        direction="KD_pH7.4_over_KD_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="ipi_acidic_ph_selective_2022",
        source=PH_SWITCH_SOURCE_IPI,
        variant_id="Ipi.64",
        mutation_signature="H31;E27a;D30",
        histidine_count=1,
        acidic_count=2,
        mutation_count=3,
        ratio=24.1,
        direction="KD_pH7.4_over_KD_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="ipi_acidic_ph_selective_2022",
        source=PH_SWITCH_SOURCE_IPI,
        variant_id="Ipi.71",
        mutation_signature="H31;H55;E27a;D30",
        histidine_count=2,
        acidic_count=2,
        mutation_count=4,
        ratio=45.5,
        direction="KD_pH7.4_over_KD_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="ipi_acidic_ph_selective_2022",
        source=PH_SWITCH_SOURCE_IPI,
        variant_id="Ipi.95",
        mutation_signature="H31;H55;E27a;D30;E32",
        histidine_count=2,
        acidic_count=3,
        mutation_count=5,
        ratio=135.0,
        direction="KD_pH7.4_over_KD_pH6.0",
        censored_lower_bound=True,
    ),
    _public_ph_switch_row(
        dataset_id="ipi_acidic_ph_selective_2022",
        source=PH_SWITCH_SOURCE_IPI,
        variant_id="Ipi.92",
        mutation_signature="H31;H55;H95;E27a;D30;E32",
        histidine_count=3,
        acidic_count=3,
        mutation_count=6,
        ratio=19.2,
        direction="KD_pH7.4_over_KD_pH6.0",
        censored_lower_bound=True,
    ),
    # Anti-CEA acidic-pH binding variants from purified-mutant EC50 values.
    # The reported ratio is EC50 pH7.4 / EC50 pH6.0.
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="Wild-type",
        mutation_signature="WT",
        histidine_count=0,
        acidic_count=0,
        mutation_count=0,
        ratio=1.45,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="LF36D",
        mutation_signature="F36D",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=29.36,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="LF36E",
        mutation_signature="F36E",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=32.86,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="LR54D",
        mutation_signature="R54D",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=2.66,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="LR54E",
        mutation_signature="R54E",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=4.60,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="LT95D",
        mutation_signature="T95D",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=11.34,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="HP98D",
        mutation_signature="P98D",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=11.52,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="HG100D",
        mutation_signature="G100D",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=3.52,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="HA107D",
        mutation_signature="A107D",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=7.43,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    _public_ph_switch_row(
        dataset_id="cea_acidic_binding_2022",
        source=PH_SWITCH_SOURCE_CEA,
        variant_id="HA107E",
        mutation_signature="A107E",
        histidine_count=0,
        acidic_count=1,
        mutation_count=1,
        ratio=9.49,
        direction="EC50_pH7.4_over_EC50_pH6.0",
    ),
    # Eculizumab-derived C5 release variants. The reported ratio is acidic
    # release selectivity: KD pH6.0 / KD pH7.4.
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="Ecu DS",
        mutation_signature="WT",
        histidine_count=0,
        acidic_count=0,
        mutation_count=0,
        ratio=21.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="mAb 1",
        mutation_signature="WT",
        histidine_count=0,
        acidic_count=0,
        mutation_count=0,
        ratio=24.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="mAb 2",
        mutation_signature="Y27H;L52H",
        histidine_count=2,
        acidic_count=0,
        mutation_count=2,
        ratio=35.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="mAb 3",
        mutation_signature="Y27H;S57H",
        histidine_count=2,
        acidic_count=0,
        mutation_count=2,
        ratio=8151.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="mAb 4",
        mutation_signature="I34H;S57H",
        histidine_count=2,
        acidic_count=0,
        mutation_count=2,
        ratio=68.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="mAb 5",
        mutation_signature="G31H",
        histidine_count=1,
        acidic_count=0,
        mutation_count=1,
        ratio=5758.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="mAb 6",
        mutation_signature="G31H;S57H",
        histidine_count=2,
        acidic_count=0,
        mutation_count=2,
        ratio=2770.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="alxn1210_c5_release_2018",
        source=PH_SWITCH_SOURCE_ALXN1210,
        variant_id="mAb 9",
        mutation_signature="G31H;I34H;S57H",
        histidine_count=3,
        acidic_count=0,
        mutation_count=3,
        ratio=4093.0,
        direction="KD_pH6.0_over_KD_pH7.4",
    ),
    # HER2 bH1 Fab pH-dependent binding variants from a public SPR table.
    # The reported ratio is acidic-release selectivity: KD pH5.0 / KD pH7.4.
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1",
        mutation_signature="WT",
        histidine_count=0,
        acidic_count=0,
        mutation_count=0,
        ratio=13.0 / 3.0,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1-P1",
        mutation_signature="N28H",
        histidine_count=1,
        acidic_count=0,
        mutation_count=1,
        ratio=16.0 / 3.5,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1-P2",
        mutation_signature="Y33H",
        histidine_count=1,
        acidic_count=0,
        mutation_count=1,
        ratio=1200.0 / 120.0,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1-P5",
        mutation_signature="R58H",
        histidine_count=1,
        acidic_count=0,
        mutation_count=1,
        ratio=98.0 / 310.0,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1-P7",
        mutation_signature="R30H",
        histidine_count=1,
        acidic_count=0,
        mutation_count=1,
        ratio=17.0 / 5.7,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1-P8",
        mutation_signature="S30H",
        histidine_count=1,
        acidic_count=0,
        mutation_count=1,
        ratio=9.9 / 3.4,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1-P5P7",
        mutation_signature="R58H;R30H",
        histidine_count=2,
        acidic_count=0,
        mutation_count=2,
        ratio=90.0 / 530.0,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    _public_ph_switch_row(
        dataset_id="her2_bh1_fab_release_2019",
        source=PH_SWITCH_SOURCE_HER2_BH1,
        variant_id="bH1-P5P8",
        mutation_signature="R58H;S30H",
        histidine_count=2,
        acidic_count=0,
        mutation_count=2,
        ratio=50.0 / 290.0,
        direction="KD_pH5.0_over_KD_pH7.4",
    ),
    # Adalimumab-derived rhTNF release variants from combinatorial histidine
    # scanning. The reported ratio is acidic-release selectivity:
    # kd pH6.0 / kd pH7.4.
    _public_ph_switch_row(
        dataset_id="adalimumab_tnf_release_2015",
        source=PH_SWITCH_SOURCE_ADALIMUMAB_TNF,
        variant_id="adalimumab",
        mutation_signature="WT",
        histidine_count=0,
        acidic_count=0,
        mutation_count=0,
        ratio=9.0,
        direction="KD_pH6.0_over_KD_pH7.4_release",
    ),
    _public_ph_switch_row(
        dataset_id="adalimumab_tnf_release_2015",
        source=PH_SWITCH_SOURCE_ADALIMUMAB_TNF,
        variant_id="PSV#1",
        mutation_signature="S100bH;S100cH;R90H;N92H;T97H",
        histidine_count=5,
        acidic_count=0,
        mutation_count=5,
        ratio=231.0,
        direction="KD_pH6.0_over_KD_pH7.4_release",
    ),
    _public_ph_switch_row(
        dataset_id="adalimumab_tnf_release_2015",
        source=PH_SWITCH_SOURCE_ADALIMUMAB_TNF,
        variant_id="PSV#2",
        mutation_signature="S100bH;S100cH;Q89H;R90H;N92H",
        histidine_count=5,
        acidic_count=0,
        mutation_count=5,
        ratio=785.0,
        direction="KD_pH6.0_over_KD_pH7.4_release",
    ),
    _public_ph_switch_row(
        dataset_id="adalimumab_tnf_release_2015",
        source=PH_SWITCH_SOURCE_ADALIMUMAB_TNF,
        variant_id="PSV#3",
        mutation_signature="L98H;Y32H;L33H",
        histidine_count=3,
        acidic_count=0,
        mutation_count=3,
        ratio=505.0,
        direction="KD_pH6.0_over_KD_pH7.4_release",
    ),
)

RESULT_COLUMNS = (
    "status",
    "error",
    "dataset_id",
    "source",
    "seed",
    "method",
    "budget",
    "train_count",
    "test_count",
    "selected_count",
    "selected_ids",
    "best_selected_fitness",
    "best_selected_normalized_fitness",
    "mean_selected_fitness",
    "hit_rate_top_decile",
    "regret_vs_oracle",
    "selector_uses_oracle_truth",
    "split_protocol",
)

SUMMARY_COLUMNS = (
    "method",
    "dataset_id",
    "source",
    "replicate_count",
    "dataset_count",
    "mean_best_selected_fitness",
    "mean_best_selected_normalized_fitness",
    "mean_hit_rate_top_decile",
    "mean_regret_vs_oracle",
    "selector_uses_oracle_truth",
    "interpretation",
)

PH_SWITCH_RESULT_COLUMNS = (
    "status",
    "error",
    "dataset_id",
    "source",
    "method",
    "budget",
    "variant_count",
    "selected_count",
    "selected_ids",
    "best_selected_ph_ratio",
    "best_selected_normalized_log_ratio",
    "mean_selected_ph_ratio",
    "hit_rate_top_tertile",
    "regret_vs_oracle",
    "selector_uses_oracle_truth",
    "interpretation",
)

PH_SWITCH_SUMMARY_COLUMNS = (
    "method",
    "dataset_id",
    "source",
    "replicate_count",
    "dataset_count",
    "mean_best_selected_ph_ratio",
    "mean_best_selected_normalized_log_ratio",
    "mean_hit_rate_top_tertile",
    "mean_regret_vs_oracle",
    "selector_uses_oracle_truth",
    "interpretation",
)

PH_SWITCH_VARIANT_REPLAY_COLUMNS = (
    "dataset_id",
    "source",
    "variant_id",
    "mutation_signature",
    "histidine_count",
    "acidic_count",
    "mutation_count",
    "observed_ph_ratio",
    "observed_log_ratio",
    "observed_normalized_log_ratio",
    "predicted_score",
    "transition_context_score",
    "ph_switch_prior_score",
    "histidine_count_score",
    "ionizable_count_score",
    "actual_top_tertile",
    "model_training_row_count",
)

PH_SWITCH_VARIANT_METHOD_SUMMARY_COLUMNS = (
    "method",
    "variant_count",
    "dataset_count",
    "score_column",
    "selector_inputs_exclude_heldout_reported_pH_ratio",
    "predicted_vs_observed_normalized_log_ratio_pearson",
    "predicted_vs_observed_normalized_log_ratio_spearman",
    "top_tertile_base_rate",
    "top_tertile_hit_rate_at_top_third_by_score",
    "top_tertile_enrichment_at_top_third_by_score",
    "interpretation",
)


@dataclass(frozen=True)
class ExternalRecord:
    variant_id: str
    dataset_id: str
    source: str
    heavy: str
    light: str
    fitness: float
    normalized_fitness: float
    split_value: str
    design_label: str


def run_external_ph_switch_benchmark(
    root: str | Path,
    *,
    run_id: str = DEFAULT_PH_SWITCH_RUN_ID,
    rows: Sequence[Mapping[str, Any]] | None = None,
    budget: int = DEFAULT_PH_SWITCH_BUDGET,
) -> dict[str, Any]:
    """Run a small public pH-switch residue-prior sanity benchmark.

    This benchmark is intentionally narrow. It does not validate generated 1E62
    candidates; it checks whether the pH-switch residue-class prior used by the
    mechanism ranks published acidic pH-selective antibody variants above their
    parental antibody without using the reported pH ratio as selector input.
    """

    project_root = Path(root).expanduser().resolve()
    run_dir = project_root / "runs" / _safe_run_id(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    if budget <= 0:
        raise ValueError("budget must be positive")
    records = [_normalize_ph_switch_row(row) for row in (rows or DEFAULT_PUBLIC_PH_SWITCH_ROWS)]
    if len(records) < 4:
        raise ValueError("external pH-switch benchmark requires at least four rows")
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record["dataset_id"])].append(record)

    result_rows: list[dict[str, Any]] = []
    for dataset_id, dataset_rows in sorted(by_dataset.items()):
        result_rows.extend(_evaluate_ph_switch_dataset(dataset_id, dataset_rows, budget=budget))
    transfer_rows, transfer_trace, transfer_model = _evaluate_ph_switch_literature_transfer(
        records,
        budget=budget,
    )
    result_rows.extend(transfer_rows)
    variant_replay_rows, variant_replay_summary = _evaluate_ph_switch_variant_replay(records)

    summary_rows = _summary_ph_switch_rows(result_rows)
    benchmark_path = run_dir / "external_ph_switch_benchmark_results.csv"
    summary_path = run_dir / "external_ph_switch_benchmark_summary.csv"
    variant_replay_path = run_dir / "external_ph_switch_variant_replay_results.csv"
    variant_replay_summary_path = run_dir / "external_ph_switch_variant_replay_summary.json"
    variant_method_summary_path = run_dir / "external_ph_switch_variant_method_summary.csv"
    config_path = run_dir / "external_ph_switch_benchmark_config.json"
    trace_path = run_dir / "external_ph_switch_source_trace.json"
    claims_path = run_dir / "external_ph_switch_claims.json"
    transfer_model_path = run_dir / "external_ph_switch_literature_transfer_model.json"
    raw_path = run_dir / "external_ph_switch_curated_records.csv"
    _write_csv(benchmark_path, result_rows, PH_SWITCH_RESULT_COLUMNS)
    _write_csv(summary_path, summary_rows, PH_SWITCH_SUMMARY_COLUMNS)
    _write_csv(variant_replay_path, variant_replay_rows, PH_SWITCH_VARIANT_REPLAY_COLUMNS)
    _write_csv(
        variant_method_summary_path,
        variant_replay_summary.get("method_summaries", []),
        PH_SWITCH_VARIANT_METHOD_SUMMARY_COLUMNS,
    )
    _write_json(variant_replay_summary_path, variant_replay_summary)
    _write_csv(
        raw_path,
        records,
        (
            "dataset_id",
            "source",
            "variant_id",
            "mutation_signature",
            "histidine_count",
            "acidic_count",
            "mutation_count",
            "ph74_over_ph60_kd_ratio",
            "selectivity_direction",
            "normalized_log_ratio",
            "censored_lower_bound",
        ),
    )
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "benchmark": "external_ph_switch_residue_prior",
            "run_id": _safe_run_id(run_id),
            "budget": budget,
            "evidence_scope": "public_pH_switch_antibody_table_sanity_check",
            "interpretation_note": (
                "Rows are curated from public pH-selectivity tables. The ratio direction is recorded per row "
                "in selectivity_direction because some studies optimize acidic binding and others optimize "
                "acidic release."
            ),
            "data_sources": sorted(
                {
                    (str(record["dataset_id"]), str(record["source"]))
                    for record in records
                }
            ),
            "leakage_controls": {
                "selector_inputs_exclude_reported_pH_ratio": True,
                "variant_replay_excludes_heldout_reported_pH_ratio": True,
                "oracle_top_measured_marked_as_upper_bound": True,
                "benchmark_tests_residue_prior_not_full_1E62_algorithm": True,
            },
        },
    )
    _write_json(
        trace_path,
        {
            "schema_version": 1,
            "sources": [
                {
                    "dataset_id": dataset_id,
                    "status": "curated_public_table",
                    "row_count": len(dataset_rows),
                    "source": dataset_rows[0].get("source", ""),
                }
                for dataset_id, dataset_rows in sorted(by_dataset.items())
            ],
            "literature_transfer": transfer_trace,
        },
    )
    _write_json(transfer_model_path, transfer_model)
    _write_json(
        claims_path,
        {
            "schema_version": 1,
            "claim": (
                "External pH-switch literature-table replay tests whether the residue-class prior ranks "
                "published pH-selective variants above the parent."
            ),
            "supported_scope": "pH_switch_residue_prior_sanity_check_when_summary_rows_support_it",
            "unsupported_scope": [
                "validated 1E62 pH 6.0 dissociation",
                "full algorithm validation on independent antibody sequences",
                "prospective wet-lab activity",
            ],
        },
    )
    return {
        "status": "completed" if summary_rows else "failed",
        "run_id": _safe_run_id(run_id),
        "run_dir": str(run_dir),
        "dataset_count": len(by_dataset),
        "benchmark_results_path": str(benchmark_path),
        "summary_results_path": str(summary_path),
        "config_path": str(config_path),
        "source_trace_path": str(trace_path),
        "claims_path": str(claims_path),
        "literature_transfer_model_path": str(transfer_model_path),
        "curated_records_path": str(raw_path),
        "variant_replay_results_path": str(variant_replay_path),
        "variant_replay_summary_path": str(variant_replay_summary_path),
        "variant_method_summary_path": str(variant_method_summary_path),
    }


def run_external_antibody_benchmark(
    root: str | Path,
    *,
    run_id: str = DEFAULT_RUN_ID,
    datasets: Sequence[Mapping[str, Any]] | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    budget: int = DEFAULT_BUDGET,
    offline_fixtures: bool = False,
    refresh: bool = False,
) -> dict[str, Any]:
    """Run held-out replay over external antibody fitness datasets."""

    project_root = Path(root).expanduser().resolve()
    run_dir = project_root / "runs" / _safe_run_id(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_specs = list(datasets or DEFAULT_FLAB_DATASETS)
    seed_values = [int(seed) for seed in seeds]
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not seed_values:
        raise ValueError("seeds must contain at least one value")

    result_rows: list[dict[str, Any]] = []
    source_trace: list[dict[str, Any]] = []
    usable_dataset_count = 0
    for spec in dataset_specs:
        try:
            raw_text, trace = _load_dataset_text(
                spec,
                run_dir=run_dir,
                offline_fixtures=offline_fixtures,
                refresh=refresh,
            )
            source_trace.append(trace)
            records = _normalize_records(raw_text, spec)
            if len(records) < 4:
                raise ValueError("dataset has fewer than four usable records")
            usable_dataset_count += 1
            for seed in seed_values:
                result_rows.extend(
                    _evaluate_dataset_seed(records, spec, seed=seed, budget=budget)
                )
        except Exception as exc:
            result_rows.append(_failed_row(spec, error=str(exc), budget=budget))
            source_trace.append(
                {
                    "dataset_id": str(spec.get("dataset_id") or "unknown"),
                    "source": str(spec.get("source") or ""),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_rows = _summary_rows(result_rows)
    benchmark_path = run_dir / "external_antibody_benchmark_results.csv"
    summary_path = run_dir / "external_antibody_benchmark_summary.csv"
    config_path = run_dir / "external_antibody_benchmark_config.json"
    source_trace_path = run_dir / "external_antibody_source_trace.json"
    claims_path = run_dir / "external_antibody_claims.json"
    _write_csv(benchmark_path, result_rows, RESULT_COLUMNS)
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    _write_json(source_trace_path, {"schema_version": 1, "sources": source_trace})
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "benchmark": "external_antibody_replay",
            "run_id": _safe_run_id(run_id),
            "budget": budget,
            "seeds": seed_values,
            "evidence_scope": "external_retrospective_antibody_replay",
            "data_sources": [
                {
                    "dataset_id": str(spec.get("dataset_id") or ""),
                    "source": str(spec.get("source") or ""),
                    "category": str(spec.get("category") or ""),
                    "url": str(spec.get("url") or ""),
                    "path": str(spec.get("path") or ""),
                    "reference": str(spec.get("reference") or ""),
                }
                for spec in dataset_specs
            ],
            "methods": sorted({str(row.get("method")) for row in result_rows if row.get("method")}),
            "leakage_controls": {
                "selector_inputs_exclude_external_fitness": True,
                "oracle_top_measured_marked_as_upper_bound": True,
                "external_fitness_used_only_after_selection": True,
            },
        },
    )
    _write_json(
        claims_path,
        {
            "schema_version": 1,
            "claim": "External FLAb-style replay tests held-out antibody affinity ranking, not pH-switch wet-lab validation.",
            "supported_scope": "external_antibody_fitness_ranking_when_summary_rows_support_it",
            "unsupported_scope": [
                "validated 1E62 pH 6.0 dissociation",
                "prospective wet-lab activity",
                "structural mechanism of generated 1E62 mutations",
            ],
        },
    )
    status = "completed" if usable_dataset_count > 0 else "failed"
    return {
        "status": status,
        "run_id": _safe_run_id(run_id),
        "run_dir": str(run_dir),
        "dataset_count": usable_dataset_count,
        "benchmark_results_path": str(benchmark_path),
        "summary_results_path": str(summary_path),
        "config_path": str(config_path),
        "source_trace_path": str(source_trace_path),
        "claims_path": str(claims_path),
        "artifact_paths": {
            "external_antibody_benchmark_results": str(benchmark_path),
            "external_antibody_benchmark_summary": str(summary_path),
            "external_antibody_benchmark_config": str(config_path),
            "external_antibody_source_trace": str(source_trace_path),
            "external_antibody_claims": str(claims_path),
        },
    }


def _evaluate_dataset_seed(
    records: Sequence[ExternalRecord],
    spec: Mapping[str, Any],
    *,
    seed: int,
    budget: int,
) -> list[dict[str, Any]]:
    train, test, split_protocol = _split_records(records, spec, seed)
    if len(train) < 2 or len(test) < 1:
        raise ValueError(f"split produced train={len(train)} test={len(test)}")
    base = _choose_base_record(records, train)
    train_for_fit = _dedupe_records([base, *train])
    candidates = [_candidate_without_truth(record, base) for record in test if record.variant_id != base.variant_id]
    if not candidates:
        raise ValueError("no held-out candidates after removing base sequence")
    truth = {record.variant_id: record for record in test}
    oracle_best = max(record.fitness for record in test)
    threshold = _top_decile_threshold(test)
    rows = []

    method_to_ids: dict[str, list[str]] = {}
    try:
        state = pcig.fit_state(
            base.heavy,
            base.light,
            [_observed_for_pcig(record) for record in train_for_fit],
            config={"max_generated_candidates": max(len(candidates), budget)},
        )
        scored = pcig.score_candidates(state, candidates)
        selected = pcig.select_panel(state, scored, budget, random.Random(seed))
        method_to_ids["pcig_external_replay"] = selected or [str(scored[0]["candidate_id"])]
    except Exception:
        method_to_ids["pcig_external_replay"] = []

    transfer_scores = _observed_edit_effect_scores(base, train_for_fit, candidates)
    method_to_ids["observed_edit_effect_transfer"] = _top_ids(transfer_scores, budget)
    method_to_ids["fewest_edits"] = _top_ids(
        {
            str(candidate["candidate_id"]): -float(candidate.get("mutation_count", 0))
            for candidate in candidates
        },
        budget,
    )
    rng = random.Random(seed)
    shuffled = [str(candidate["candidate_id"]) for candidate in candidates]
    rng.shuffle(shuffled)
    method_to_ids["random_measured"] = shuffled[:budget]
    method_to_ids["oracle_top_measured"] = [
        record.variant_id for record in sorted(test, key=lambda item: item.fitness, reverse=True)[:budget]
    ]

    for method, selected_ids in method_to_ids.items():
        rows.append(
            _result_row(
                method=method,
                dataset_id=base.dataset_id,
                source=base.source,
                seed=seed,
                budget=budget,
                train_count=len(train_for_fit),
                test_count=len(test),
                selected_ids=selected_ids,
                truth=truth,
                oracle_best=oracle_best,
                threshold=threshold,
                split_protocol=split_protocol,
                selector_uses_oracle_truth=(method == "oracle_top_measured"),
            )
        )
    return rows


def _result_row(
    *,
    method: str,
    dataset_id: str,
    source: str,
    seed: int,
    budget: int,
    train_count: int,
    test_count: int,
    selected_ids: Sequence[str],
    truth: Mapping[str, ExternalRecord],
    oracle_best: float,
    threshold: float,
    split_protocol: str,
    selector_uses_oracle_truth: bool,
) -> dict[str, Any]:
    selected_records = [truth[item] for item in selected_ids if item in truth]
    best = max((record.fitness for record in selected_records), default=float("nan"))
    best_norm = max((record.normalized_fitness for record in selected_records), default=float("nan"))
    mean_selected = mean([record.fitness for record in selected_records]) if selected_records else float("nan")
    hit_rate = (
        mean([1.0 if record.fitness >= threshold else 0.0 for record in selected_records])
        if selected_records
        else 0.0
    )
    regret = oracle_best - best if selected_records else float("nan")
    return {
        "status": "completed" if selected_records else "failed",
        "error": "" if selected_records else "no selected IDs matched held-out measured candidates",
        "dataset_id": dataset_id,
        "source": source,
        "seed": seed,
        "method": method,
        "budget": budget,
        "train_count": train_count,
        "test_count": test_count,
        "selected_count": len(selected_records),
        "selected_ids": ";".join(record.variant_id for record in selected_records),
        "best_selected_fitness": _round(best),
        "best_selected_normalized_fitness": _round(best_norm),
        "mean_selected_fitness": _round(mean_selected),
        "hit_rate_top_decile": _round(hit_rate),
        "regret_vs_oracle": _round(regret),
        "selector_uses_oracle_truth": selector_uses_oracle_truth,
        "split_protocol": split_protocol,
    }


def _split_records(
    records: Sequence[ExternalRecord],
    spec: Mapping[str, Any],
    seed: int,
) -> tuple[list[ExternalRecord], list[ExternalRecord], str]:
    split_column = str(spec.get("split_column") or "")
    train_values = {str(value) for value in spec.get("train_values", [])}
    test_values = {str(value) for value in spec.get("test_values", [])}
    if split_column and train_values and test_values:
        train = [record for record in records if record.split_value in train_values]
        test = [record for record in records if record.split_value in test_values]
        if train and test:
            return train, test, f"{split_column}:train={','.join(sorted(train_values))};test={','.join(sorted(test_values))}"
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    split_at = max(2, min(len(shuffled) - 1, int(len(shuffled) * 0.6)))
    return shuffled[:split_at], shuffled[split_at:], "seeded_60_40_split"


def _normalize_records(text: str, spec: Mapping[str, Any]) -> list[ExternalRecord]:
    dataset_id = str(spec.get("dataset_id") or "external_dataset")
    source = str(spec.get("source") or "external")
    split_column = str(spec.get("split_column") or "")
    raw_rows = list(csv.DictReader(text.splitlines()))
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        heavy = _sequence(row.get("heavy") or row.get("vh") or row.get("heavy_chain_seq"))
        light = _sequence(row.get("light") or row.get("vl") or row.get("light_chain_seq"))
        fitness = _float(row.get("fitness") or row.get("neg_log_Kd") or row.get("-log10( KD (M) )"))
        if not heavy or fitness is None:
            continue
        parsed.append(
            {
                "index": index,
                "heavy": heavy,
                "light": light,
                "fitness": fitness,
                "split_value": str(row.get(split_column) or "") if split_column else "",
                "design_label": str(row.get("Design") or row.get("Mutations") or row.get("Method") or f"row_{index}"),
            }
        )
    if not parsed:
        return []
    min_fitness = min(item["fitness"] for item in parsed)
    max_fitness = max(item["fitness"] for item in parsed)
    span = max(max_fitness - min_fitness, 1e-12)
    records = []
    for item in parsed:
        records.append(
            ExternalRecord(
                variant_id=f"{dataset_id}:{item['index']}",
                dataset_id=dataset_id,
                source=source,
                heavy=item["heavy"],
                light=item["light"],
                fitness=float(item["fitness"]),
                normalized_fitness=(float(item["fitness"]) - min_fitness) / span,
                split_value=item["split_value"],
                design_label=item["design_label"],
            )
        )
    return records


def _load_dataset_text(
    spec: Mapping[str, Any],
    *,
    run_dir: Path,
    offline_fixtures: bool,
    refresh: bool,
) -> tuple[str, dict[str, Any]]:
    dataset_id = _safe_run_id(str(spec.get("dataset_id") or "external_dataset"))
    if spec.get("path"):
        path = Path(str(spec["path"])).expanduser().resolve()
        return path.read_text(encoding="utf-8"), {
            "dataset_id": dataset_id,
            "source": str(spec.get("source") or ""),
            "status": "ok",
            "cache_hit": False,
            "path": str(path),
        }
    cache_dir = run_dir / "external_antibody_raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_id}.csv"
    if offline_fixtures:
        cache_path.write_text(_offline_fixture_csv(), encoding="utf-8")
        return cache_path.read_text(encoding="utf-8"), {
            "dataset_id": dataset_id,
            "source": str(spec.get("source") or "fixture"),
            "status": "ok",
            "cache_hit": False,
            "offline_fixture": True,
            "cache_path": str(cache_path),
        }
    if cache_path.is_file() and not refresh:
        return cache_path.read_text(encoding="utf-8"), {
            "dataset_id": dataset_id,
            "source": str(spec.get("source") or ""),
            "status": "ok",
            "cache_hit": True,
            "cache_path": str(cache_path),
        }
    url = str(spec.get("url") or "")
    if not url:
        raise ValueError("dataset spec requires path, url, or offline_fixtures=True")
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return text, {
        "dataset_id": dataset_id,
        "source": str(spec.get("source") or ""),
        "status": "ok",
        "cache_hit": False,
        "network_fetch": True,
        "url": url,
        "cache_path": str(cache_path),
    }


def _candidate_without_truth(record: ExternalRecord, base: ExternalRecord) -> dict[str, Any]:
    edits = _edit_tokens(base, record)
    return {
        "candidate_id": record.variant_id,
        "variant_id": record.variant_id,
        "heavy_chain_seq": record.heavy,
        "light_chain_seq": record.light,
        "edit_tokens": edits,
        "operator": "external_measured_pool_candidate",
        "cost": 1.0,
        "mutation_count": len(edits),
        "design_label": record.design_label,
    }


def _observed_for_pcig(record: ExternalRecord) -> dict[str, Any]:
    return {
        "variant_id": record.variant_id,
        "heavy_chain_seq": record.heavy,
        "light_chain_seq": record.light,
        "utility": record.normalized_fitness,
        "observed_utility": record.normalized_fitness,
        "expression": 1.0,
    }


def _observed_edit_effect_scores(
    base: ExternalRecord,
    train: Sequence[ExternalRecord],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in train:
        edits = _edit_tokens(base, record)
        if not edits:
            continue
        delta = (record.normalized_fitness - base.normalized_fitness) / len(edits)
        for edit in edits:
            values[edit].append(delta)
    means = {edit: mean(items) for edit, items in values.items()}
    scores = {}
    for candidate in candidates:
        edits = [str(item) for item in candidate.get("edit_tokens", [])]
        scores[str(candidate["candidate_id"])] = sum(means.get(edit, 0.0) for edit in edits)
    return scores


def _edit_tokens(base: ExternalRecord, record: ExternalRecord) -> list[str]:
    edits: list[str] = []
    for chain, base_seq, seq in (("H", base.heavy, record.heavy), ("L", base.light, record.light)):
        for index, (from_residue, to_residue) in enumerate(zip(base_seq, seq), start=1):
            if from_residue != to_residue:
                edits.append(f"{chain}{index}{to_residue}")
    return edits


def _choose_base_record(
    records: Sequence[ExternalRecord],
    train: Sequence[ExternalRecord],
) -> ExternalRecord:
    for record in [*train, *records]:
        label = record.design_label.lower()
        if label in {"wt", "wildtype", "wild-type", "s309", "medi8825"}:
            return record
    counts = Counter((record.heavy, record.light) for record in records)
    heavy, light = counts.most_common(1)[0][0]
    for record in records:
        if record.heavy == heavy and record.light == light:
            return record
    return records[0]


def _dedupe_records(records: Sequence[ExternalRecord]) -> list[ExternalRecord]:
    seen = set()
    out = []
    for record in records:
        if record.variant_id in seen:
            continue
        seen.add(record.variant_id)
        out.append(record)
    return out


def _top_decile_threshold(records: Sequence[ExternalRecord]) -> float:
    values = sorted((record.fitness for record in records), reverse=True)
    index = max(0, min(len(values) - 1, int(len(values) * 0.1)))
    return values[index]


def _top_ids(scores: Mapping[str, float], budget: int) -> list[str]:
    return [
        key
        for key, _value in sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[
            :budget
        ]
    ]


def _summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(str(row.get("method")), str(row.get("dataset_id")))].append(row)
        grouped[(str(row.get("method")), "overall")].append(row)
    out: list[dict[str, Any]] = []
    for (method, dataset_id), items in sorted(grouped.items()):
        dataset_ids = {str(row.get("dataset_id")) for row in items}
        source_values = sorted({str(row.get("source")) for row in items if row.get("source")})
        out.append(
            {
                "method": method,
                "dataset_id": dataset_id,
                "source": ";".join(source_values),
                "replicate_count": len(items),
                "dataset_count": len(dataset_ids),
                "mean_best_selected_fitness": _round(
                    mean(_float(row.get("best_selected_fitness")) or 0.0 for row in items)
                ),
                "mean_best_selected_normalized_fitness": _round(
                    mean(_float(row.get("best_selected_normalized_fitness")) or 0.0 for row in items)
                ),
                "mean_hit_rate_top_decile": _round(
                    mean(_float(row.get("hit_rate_top_decile")) or 0.0 for row in items)
                ),
                "mean_regret_vs_oracle": _round(
                    mean(_float(row.get("regret_vs_oracle")) or 0.0 for row in items)
                ),
                "selector_uses_oracle_truth": any(
                    str(row.get("selector_uses_oracle_truth")) == "True" for row in items
                ),
                "interpretation": _summary_interpretation(method),
            }
        )
    return out


def _summary_interpretation(method: str) -> str:
    if method == "pcig_external_replay":
        return "PCIG scoring over held-out externally measured antibody variants"
    if method == "observed_edit_effect_transfer":
        return "simple additive transfer of observed edit effects from train split"
    if method == "fewest_edits":
        return "low-mutation-count baseline"
    if method == "random_measured":
        return "random held-out measured-candidate baseline"
    if method == "oracle_top_measured":
        return "upper bound using held-out external fitness after selection"
    return "external replay method"


def _normalize_ph_switch_row(row: Mapping[str, Any]) -> dict[str, Any]:
    ratio = _float(row.get("ph74_over_ph60_kd_ratio"))
    if ratio is None or ratio <= 0:
        raise ValueError("pH-switch row requires a positive ph74_over_ph60_kd_ratio")
    log_ratio = math.log10(ratio)
    return {
        "dataset_id": str(row.get("dataset_id") or "public_ph_switch"),
        "source": str(row.get("source") or ""),
        "variant_id": str(row.get("variant_id") or ""),
        "mutation_signature": str(row.get("mutation_signature") or ""),
        "histidine_count": int(row.get("histidine_count") or 0),
        "acidic_count": int(row.get("acidic_count") or 0),
        "mutation_count": int(row.get("mutation_count") or 0),
        "ph74_over_ph60_kd_ratio": ratio,
        "selectivity_direction": str(row.get("selectivity_direction") or "not_recorded"),
        "normalized_log_ratio": log_ratio,
        "censored_lower_bound": bool(row.get("censored_lower_bound")),
    }


def _evaluate_ph_switch_dataset(
    dataset_id: str,
    records: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> list[dict[str, Any]]:
    normalized_records = _normalize_ph_switch_dataset_records(records)
    threshold = _top_tertile_ph_ratio_threshold(normalized_records)
    source = str(normalized_records[0].get("source") or "")
    method_scores = _ph_switch_blind_method_scores(normalized_records)
    method_to_ids = {
        method: _top_ids(scores, budget)
        for method, scores in method_scores.items()
    }
    method_to_ids["oracle_top_pH_ratio"] = [
        str(record["variant_id"])
        for record in sorted(
            normalized_records,
            key=lambda item: (float(item["ph74_over_ph60_kd_ratio"]), str(item["variant_id"])),
            reverse=True,
        )[:budget]
    ]
    method_to_ids["parent_reference"] = [
        str(record["variant_id"])
        for record in normalized_records
        if int(record.get("mutation_count") or 0) == 0
    ][:budget]
    return [
        _ph_switch_result_row(
            method=method,
            dataset_id=dataset_id,
            source=source,
            budget=budget,
            variant_count=len(normalized_records),
            selected_ids=selected_ids,
            records=normalized_records,
            oracle_best=max(float(record["ph74_over_ph60_kd_ratio"]) for record in normalized_records),
            threshold=threshold,
            selector_uses_oracle_truth=(method == "oracle_top_pH_ratio"),
        )
        for method, selected_ids in method_to_ids.items()
    ]


def _normalize_ph_switch_dataset_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    min_log = min(float(record["normalized_log_ratio"]) for record in records)
    max_log = max(float(record["normalized_log_ratio"]) for record in records)
    span = max(max_log - min_log, 1e-12)
    normalized_records = []
    for record in records:
        item = dict(record)
        item["normalized_log_ratio"] = (float(record["normalized_log_ratio"]) - min_log) / span
        normalized_records.append(item)
    return normalized_records


def _top_tertile_ph_ratio_threshold(records: Sequence[Mapping[str, Any]]) -> float:
    return sorted(
        (float(record["ph74_over_ph60_kd_ratio"]) for record in records),
        reverse=True,
    )[max(0, min(len(records) - 1, int(len(records) / 3) - 1))]


def _ph_switch_blind_method_scores(
    normalized_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    return {
        "ph_switch_residue_prior": {
            str(record["variant_id"]): _ph_switch_prior_score(record)
            for record in normalized_records
        },
        "histidine_count": {
            str(record["variant_id"]): float(record.get("histidine_count") or 0)
            for record in normalized_records
        },
        "ionizable_count": {
            str(record["variant_id"]): float(record.get("histidine_count") or 0)
            + float(record.get("acidic_count") or 0)
            for record in normalized_records
        },
        "transition_context_prior": {
            str(record["variant_id"]): _transition_context_prior_score(record)
            for record in normalized_records
        },
        "fewest_edits": {
            str(record["variant_id"]): -float(record.get("mutation_count") or 0)
            for record in normalized_records
        },
    }


def _evaluate_ph_switch_literature_transfer(
    records: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Leave-one-study replay for literature-calibrated transition features.

    Each held-out public pH-switch table is scored using feature weights learned
    from the other tables. Reported pH ratios from the held-out table remain
    hidden until evaluation.
    """

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record["dataset_id"])].append(dict(record))
    normalized_by_dataset = {
        dataset_id: _normalize_ph_switch_dataset_records(dataset_rows)
        for dataset_id, dataset_rows in sorted(by_dataset.items())
    }
    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    model_snapshots: dict[str, Any] = {
        "schema_version": 1,
        "method": "leave_one_study_transition_calibration",
        "selector_inputs_exclude_heldout_reported_pH_ratio": True,
        "heldout_models": {},
    }
    for heldout_id, heldout_rows in normalized_by_dataset.items():
        train_rows = [
            row
            for dataset_id, dataset_rows in normalized_by_dataset.items()
            if dataset_id != heldout_id
            for row in dataset_rows
        ]
        if not train_rows:
            continue
        model = _learn_literature_transition_model(train_rows)
        scores = {
            str(record["variant_id"]): _literature_transition_model_score(record, model)
            for record in heldout_rows
        }
        selected_ids = _top_ids(scores, budget)
        source = str(heldout_rows[0].get("source") or "")
        result_rows.append(
            _ph_switch_result_row(
                method="leave_one_study_transition_calibration",
                dataset_id=heldout_id,
                source=source,
                budget=budget,
                variant_count=len(heldout_rows),
                selected_ids=selected_ids,
                records=heldout_rows,
                oracle_best=max(float(record["ph74_over_ph60_kd_ratio"]) for record in heldout_rows),
                threshold=_top_tertile_ph_ratio_threshold(heldout_rows),
                selector_uses_oracle_truth=False,
            )
        )
        trace_rows.append(
            {
                "heldout_dataset_id": heldout_id,
                "training_dataset_ids": sorted(dataset_id for dataset_id in normalized_by_dataset if dataset_id != heldout_id),
                "training_row_count": len(train_rows),
                "heldout_row_count": len(heldout_rows),
                "selected_ids": selected_ids,
                "feature_count": len(model["feature_weights"]),
            }
        )
        model_snapshots["heldout_models"][heldout_id] = model
    return result_rows, trace_rows, model_snapshots


def _evaluate_ph_switch_variant_replay(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Leave one measured public pH-switch variant out and predict its rank.

    The learned score uses all other public rows and the held-out variant's
    mutation description. The held-out pH ratio is added only after scoring for
    correlation and top-tertile evaluation.
    """

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        item = dict(record)
        item["observed_log_ratio"] = float(record["normalized_log_ratio"])
        by_dataset[str(record["dataset_id"])].append(item)
    normalized_records: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for dataset_id, dataset_rows in sorted(by_dataset.items()):
        normalized_rows = _normalize_ph_switch_dataset_records(dataset_rows)
        thresholds[dataset_id] = _top_tertile_ph_ratio_threshold(normalized_rows)
        normalized_records.extend(normalized_rows)
    rows: list[dict[str, Any]] = []
    for index, heldout in enumerate(normalized_records):
        train_rows = [
            row
            for train_index, row in enumerate(normalized_records)
            if train_index != index
        ]
        model = _learn_literature_transition_model(train_rows)
        predicted_score = _literature_transition_model_score(heldout, model)
        dataset_id = str(heldout["dataset_id"])
        rows.append(
            {
                "dataset_id": dataset_id,
                "source": str(heldout.get("source") or ""),
                "variant_id": str(heldout["variant_id"]),
                "mutation_signature": str(heldout.get("mutation_signature") or ""),
                "histidine_count": int(heldout.get("histidine_count") or 0),
                "acidic_count": int(heldout.get("acidic_count") or 0),
                "mutation_count": int(heldout.get("mutation_count") or 0),
                "observed_ph_ratio": _round(heldout.get("ph74_over_ph60_kd_ratio")),
                "observed_log_ratio": _round(heldout.get("observed_log_ratio")),
                "observed_normalized_log_ratio": _round(heldout.get("normalized_log_ratio")),
                "predicted_score": _round(predicted_score),
                "transition_context_score": _round(_transition_context_prior_score(heldout)),
                "ph_switch_prior_score": _round(_ph_switch_prior_score(heldout)),
                "histidine_count_score": _round(heldout.get("histidine_count")),
                "ionizable_count_score": _round(
                    float(heldout.get("histidine_count") or 0.0)
                    + float(heldout.get("acidic_count") or 0.0)
                ),
                "actual_top_tertile": (
                    float(heldout["ph74_over_ph60_kd_ratio"]) >= thresholds[dataset_id]
                ),
                "model_training_row_count": len(train_rows),
            }
        )
    summary = _ph_switch_variant_replay_summary(rows)
    return rows, summary


def _ph_switch_variant_replay_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    method_rows = _ph_switch_variant_method_summary_rows(rows)
    primary = next(
        (
            row
            for row in method_rows
            if row.get("method") == "leave_one_variant_transition_calibration"
        ),
        {},
    )
    scores = [_float(row.get("predicted_score")) for row in rows]
    observed = [_float(row.get("observed_normalized_log_ratio")) for row in rows]
    paired = [
        (float(score), float(value))
        for score, value in zip(scores, observed, strict=False)
        if score is not None and value is not None
    ]
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            _float(row.get("predicted_score")) if _float(row.get("predicted_score")) is not None else -math.inf,
            str(row.get("dataset_id") or ""),
            str(row.get("variant_id") or ""),
        ),
        reverse=True,
    )
    top_k = max(1, math.ceil(len(ranked_rows) / 3))
    top_rows = ranked_rows[:top_k]
    base_rate = mean([1.0 if row.get("actual_top_tertile") else 0.0 for row in rows]) if rows else 0.0
    top_rate = mean([1.0 if row.get("actual_top_tertile") else 0.0 for row in top_rows]) if top_rows else 0.0
    dataset_ids = sorted({str(row.get("dataset_id") or "") for row in rows if row.get("dataset_id")})
    return {
        "schema_version": 1,
        "method": "leave_one_variant_transition_calibration",
        "variant_count": len(rows),
        "dataset_count": len(dataset_ids),
        "selector_inputs_exclude_heldout_reported_pH_ratio": True,
        "predicted_vs_observed_normalized_log_ratio_pearson": _round_number(
            primary.get("predicted_vs_observed_normalized_log_ratio_pearson")
            if primary
            else _pearson_correlation([item[0] for item in paired], [item[1] for item in paired])
        ),
        "predicted_vs_observed_normalized_log_ratio_spearman": _round_number(
            primary.get("predicted_vs_observed_normalized_log_ratio_spearman")
            if primary
            else _pearson_correlation(
                    _rank_values([item[0] for item in paired]),
                    _rank_values([item[1] for item in paired]),
                )
        ),
        "top_tertile_base_rate": _round_number(
            primary.get("top_tertile_base_rate") if primary else base_rate
        ),
        "top_tertile_hit_rate_at_top_third_by_score": _round_number(
            primary.get("top_tertile_hit_rate_at_top_third_by_score") if primary else top_rate
        ),
        "top_tertile_enrichment_at_top_third_by_score": _round_number(
            primary.get("top_tertile_enrichment_at_top_third_by_score")
            if primary
            else (top_rate / base_rate if base_rate > 0 else float("nan"))
        ),
        "method_summaries": method_rows,
        "interpretation": (
            "Leave-one-variant public pH-switch replay; the held-out pH ratio is hidden during scoring. "
            "This is a literature-table sanity check, not 1E62 wet-lab validation."
        ),
    }


def _ph_switch_variant_method_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    method_columns = (
        (
            "leave_one_variant_transition_calibration",
            "predicted_score",
            "learned transition-feature model refit after each held-out public variant is removed",
        ),
        (
            "transition_context_prior",
            "transition_context_score",
            "fixed direction-aware mutation-context prior; no fitted pH-ratio outcome is used",
        ),
        (
            "ph_switch_residue_prior",
            "ph_switch_prior_score",
            "fixed histidine/acidic-residue synergy prior",
        ),
        (
            "histidine_count",
            "histidine_count_score",
            "simple histidine-count baseline",
        ),
        (
            "ionizable_count",
            "ionizable_count_score",
            "simple histidine-plus-acidic-residue-count baseline",
        ),
    )
    return [
        _ph_switch_variant_method_summary_row(rows, method, score_column, interpretation)
        for method, score_column, interpretation in method_columns
    ]


def _ph_switch_variant_method_summary_row(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    score_column: str,
    interpretation: str,
) -> dict[str, Any]:
    paired = [
        (float(score), float(value))
        for score, value in (
            (_float(row.get(score_column)), _float(row.get("observed_normalized_log_ratio")))
            for row in rows
        )
        if score is not None and value is not None
    ]
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            _float(row.get(score_column)) if _float(row.get(score_column)) is not None else -math.inf,
            str(row.get("dataset_id") or ""),
            str(row.get("variant_id") or ""),
        ),
        reverse=True,
    )
    top_k = max(1, math.ceil(len(ranked_rows) / 3))
    top_rows = ranked_rows[:top_k]
    base_rate = mean([1.0 if row.get("actual_top_tertile") else 0.0 for row in rows]) if rows else 0.0
    top_rate = mean([1.0 if row.get("actual_top_tertile") else 0.0 for row in top_rows]) if top_rows else 0.0
    dataset_ids = sorted({str(row.get("dataset_id") or "") for row in rows if row.get("dataset_id")})
    return {
        "method": method,
        "variant_count": len(rows),
        "dataset_count": len(dataset_ids),
        "score_column": score_column,
        "selector_inputs_exclude_heldout_reported_pH_ratio": True,
        "predicted_vs_observed_normalized_log_ratio_pearson": _round_number(
            _pearson_correlation([item[0] for item in paired], [item[1] for item in paired])
        ),
        "predicted_vs_observed_normalized_log_ratio_spearman": _round_number(
            _pearson_correlation(
                _rank_values([item[0] for item in paired]),
                _rank_values([item[1] for item in paired]),
            )
        ),
        "top_tertile_base_rate": _round_number(base_rate),
        "top_tertile_hit_rate_at_top_third_by_score": _round_number(top_rate),
        "top_tertile_enrichment_at_top_third_by_score": _round_number(
            top_rate / base_rate if base_rate > 0 else float("nan")
        ),
        "interpretation": interpretation,
    }


def _learn_literature_transition_model(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    global_mean = mean(float(record["normalized_log_ratio"]) for record in records)
    feature_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record["normalized_log_ratio"])
        for feature in _literature_transition_features(record):
            feature_values[feature].append(value)
    weights: dict[str, float] = {}
    support: dict[str, int] = {}
    for feature, values in sorted(feature_values.items()):
        shrinkage = len(values) / (len(values) + 2.0)
        weights[feature] = _round((mean(values) - global_mean) * shrinkage)
        support[feature] = len(values)
    return {
        "global_mean": _round(global_mean),
        "feature_weights": weights,
        "feature_support": support,
    }


def _literature_transition_model_score(
    record: Mapping[str, Any],
    model: Mapping[str, Any],
) -> float:
    weights = model.get("feature_weights") if isinstance(model.get("feature_weights"), Mapping) else {}
    score = float(model.get("global_mean") or 0.0)
    for feature in _literature_transition_features(record):
        score += float(weights.get(feature, 0.0))
    score += 0.03 * _transition_context_prior_score(record)
    score -= 0.04 * max(float(record.get("mutation_count") or 0.0) - 4.0, 0.0)
    return score


def _literature_transition_features(record: Mapping[str, Any]) -> tuple[str, ...]:
    features: list[str] = []
    direction = str(record.get("selectivity_direction") or "").lower()
    release_like = "ph6.0_over" in direction or "ph5.0_over" in direction or "release" in direction
    features.append("direction:release_like" if release_like else "direction:acidic_binding_like")
    histidine = int(record.get("histidine_count") or 0)
    acidic = int(record.get("acidic_count") or 0)
    mutation_count = int(record.get("mutation_count") or 0)
    features.append(f"histidine_bin:{min(histidine, 3)}")
    features.append(f"acidic_bin:{min(acidic, 3)}")
    features.append(f"mutation_bin:{min(mutation_count, 6)}")
    if histidine and acidic:
        features.append("mixed_histidine_acidic")
    for from_residue, _position, to_residue in _mutation_transitions(str(record.get("mutation_signature") or "")):
        if to_residue:
            features.append(f"to:{to_residue}")
        if from_residue and to_residue:
            features.append(f"transition:{from_residue}>{to_residue}")
            features.append(f"from_class:{_residue_class(from_residue)}:to:{to_residue}")
            features.append(f"from:{from_residue}:to_class:{_residue_class(to_residue)}")
    return tuple(dict.fromkeys(features))


def _residue_class(residue: str) -> str:
    value = residue.upper()
    if value == "H":
        return "histidine"
    if value in {"D", "E"}:
        return "acidic"
    if value in {"K", "R"}:
        return "basic"
    if value in {"S", "T", "N", "Q", "Y"}:
        return "polar"
    if value in {"F", "W"}:
        return "aromatic"
    if value in {"A", "V", "I", "L", "M", "G"}:
        return "hydrophobic"
    return "other"


def _ph_switch_prior_score(record: Mapping[str, Any]) -> float:
    histidine = float(record.get("histidine_count") or 0)
    acidic = float(record.get("acidic_count") or 0)
    mutation_count = float(record.get("mutation_count") or 0)
    synergy = min(histidine, acidic)
    return 1.20 * histidine + 0.80 * acidic + 0.55 * synergy - 0.08 * max(mutation_count - 4.0, 0.0)


def _transition_context_prior_score(record: Mapping[str, Any]) -> float:
    """Score public pH-switch rows using direction-aware transition context.

    This is still truth-blind: the reported pH ratio is not used. The score uses
    assay direction plus mutation signatures to distinguish a pH-release
    histidine switch from an acidic-pH binding mutation. That makes it a more
    demanding external sanity check than raw histidine/ionizable counts.
    """

    histidine = float(record.get("histidine_count") or 0)
    acidic = float(record.get("acidic_count") or 0)
    mutation_count = float(record.get("mutation_count") or 0)
    direction = str(record.get("selectivity_direction") or "").lower()
    mutations = _mutation_transitions(str(record.get("mutation_signature") or ""))
    release_like = "ph6.0_over" in direction or "release" in direction
    if release_like:
        score = 1.18 * histidine - 1.20 * max(histidine - 2.0, 0.0) ** 2 - 0.16 * acidic
    else:
        score = 1.08 * acidic + 0.38 * histidine + 0.30 * min(histidine, acidic)
        score -= 0.08 * max(mutation_count - 5.0, 0.0)
    for from_residue, _position, to_residue in mutations:
        if to_residue == "H":
            score += _histidine_transition_bonus(from_residue, release_like=release_like)
        elif to_residue in {"D", "E"}:
            score += _acidic_transition_bonus(from_residue, to_residue, release_like=release_like)
        else:
            score += 0.05 if to_residue in {"Q", "N", "S", "T", "Y"} else 0.0
    return score


def _mutation_transitions(signature: str) -> tuple[tuple[str, int | None, str], ...]:
    transitions: list[tuple[str, int | None, str]] = []
    for token in re.split(r"[;,]\s*", signature.strip()):
        if not token or token.upper() == "WT":
            continue
        match = re.fullmatch(r"([A-Z])(\d+[a-zA-Z]?)([A-Z])", token.strip())
        if match:
            from_residue, raw_position, to_residue = match.groups()
            position_match = re.search(r"\d+", raw_position)
            position = int(position_match.group(0)) if position_match else None
            transitions.append((from_residue, position, to_residue))
            continue
        to_match = re.match(r"([A-Z])", token.strip())
        if to_match:
            transitions.append(("", None, to_match.group(1)))
    return tuple(transitions)


def _histidine_transition_bonus(from_residue: str, *, release_like: bool) -> float:
    residue = from_residue.upper()
    if not release_like:
        return 0.18 if residue in {"Y", "F", "W", "S", "T", "N", "Q"} else 0.05
    if residue in {"Y", "S", "G"}:
        return 0.58
    if residue in {"L", "I", "V", "A"}:
        return 0.22
    if residue in {"D", "E", "K", "R"}:
        return -0.15
    return 0.08


def _acidic_transition_bonus(from_residue: str, to_residue: str, *, release_like: bool) -> float:
    residue = from_residue.upper()
    if release_like:
        return -0.18
    bonus = 0.0
    if residue in {"F", "Y", "W"}:
        bonus += 0.78
    elif residue in {"T", "P", "A", "S"}:
        bonus += 0.42
    elif residue in {"G", "R", "K", "H"}:
        bonus -= 0.10
    elif residue in {"N", "Q"}:
        bonus += 0.20
    if to_residue == "E" and residue in {"F", "A"}:
        bonus += 0.08
    return bonus


def _ph_switch_result_row(
    *,
    method: str,
    dataset_id: str,
    source: str,
    budget: int,
    variant_count: int,
    selected_ids: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    oracle_best: float,
    threshold: float,
    selector_uses_oracle_truth: bool,
) -> dict[str, Any]:
    truth = {str(record["variant_id"]): record for record in records}
    selected = [truth[item] for item in selected_ids if item in truth]
    best = max((float(record["ph74_over_ph60_kd_ratio"]) for record in selected), default=float("nan"))
    best_norm = max((float(record["normalized_log_ratio"]) for record in selected), default=float("nan"))
    mean_selected = (
        mean([float(record["ph74_over_ph60_kd_ratio"]) for record in selected])
        if selected
        else float("nan")
    )
    hit_rate = (
        mean([1.0 if float(record["ph74_over_ph60_kd_ratio"]) >= threshold else 0.0 for record in selected])
        if selected
        else 0.0
    )
    return {
        "status": "completed" if selected else "failed",
        "error": "" if selected else "no selected IDs matched public pH-switch records",
        "dataset_id": dataset_id,
        "source": source,
        "method": method,
        "budget": budget,
        "variant_count": variant_count,
        "selected_count": len(selected),
        "selected_ids": ";".join(str(record["variant_id"]) for record in selected),
        "best_selected_ph_ratio": _round(best),
        "best_selected_normalized_log_ratio": _round(best_norm),
        "mean_selected_ph_ratio": _round(mean_selected),
        "hit_rate_top_tertile": _round(hit_rate),
        "regret_vs_oracle": _round(oracle_best - best if selected else float("nan")),
        "selector_uses_oracle_truth": selector_uses_oracle_truth,
        "interpretation": _ph_switch_summary_interpretation(method),
    }


def _summary_ph_switch_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(str(row.get("method")), str(row.get("dataset_id")))].append(row)
        grouped[(str(row.get("method")), "overall")].append(row)
    out: list[dict[str, Any]] = []
    for (method, dataset_id), items in sorted(grouped.items()):
        dataset_ids = {str(row.get("dataset_id")) for row in items}
        source_values = sorted({str(row.get("source")) for row in items if row.get("source")})
        out.append(
            {
                "method": method,
                "dataset_id": dataset_id,
                "source": ";".join(source_values),
                "replicate_count": len(items),
                "dataset_count": len(dataset_ids),
                "mean_best_selected_ph_ratio": _round(
                    mean(_float(row.get("best_selected_ph_ratio")) or 0.0 for row in items)
                ),
                "mean_best_selected_normalized_log_ratio": _round(
                    mean(_float(row.get("best_selected_normalized_log_ratio")) or 0.0 for row in items)
                ),
                "mean_hit_rate_top_tertile": _round(
                    mean(_float(row.get("hit_rate_top_tertile")) or 0.0 for row in items)
                ),
                "mean_regret_vs_oracle": _round(
                    mean(_float(row.get("regret_vs_oracle")) or 0.0 for row in items)
                ),
                "selector_uses_oracle_truth": any(
                    str(row.get("selector_uses_oracle_truth")) == "True" for row in items
                ),
                "interpretation": _ph_switch_summary_interpretation(method),
            }
        )
    return out


def _ph_switch_summary_interpretation(method: str) -> str:
    if method == "ph_switch_residue_prior":
        return "pH-switch residue-class prior using histidine/acidic-residue synergy; not a full 1E62 algorithm validation"
    if method == "histidine_count":
        return "simple histidine-count baseline"
    if method == "ionizable_count":
        return "simple ionizable-residue-count baseline"
    if method == "transition_context_prior":
        return "direction-aware mutation-transition prior; uses residue context, not reported pH ratios"
    if method == "leave_one_study_transition_calibration":
        return "leave-one-study literature-calibrated transition model; held-out study pH ratios used only after selection"
    if method == "fewest_edits":
        return "parent-proximal baseline"
    if method == "parent_reference":
        return "unmutated parent reference"
    if method == "oracle_top_pH_ratio":
        return "upper bound using reported pH ratio after selection"
    return "external pH-switch literature-table replay method"


def _failed_row(spec: Mapping[str, Any], *, error: str, budget: int) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": error,
        "dataset_id": str(spec.get("dataset_id") or "unknown"),
        "source": str(spec.get("source") or ""),
        "seed": "",
        "method": "",
        "budget": budget,
        "train_count": "",
        "test_count": "",
        "selected_count": "",
        "selected_ids": "",
        "best_selected_fitness": "",
        "best_selected_normalized_fitness": "",
        "mean_selected_fitness": "",
        "hit_rate_top_decile": "",
        "regret_vs_oracle": "",
        "selector_uses_oracle_truth": "",
        "split_protocol": "",
    }


def _sequence(value: Any) -> str:
    text = str(value or "").strip().upper()
    return "".join(residue for residue in text if residue.isalpha())


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.6f}"


def _round_number(value: Any) -> float | None:
    number = _float(value)
    if number is None or not math.isfinite(number):
        return None
    return float(f"{number:.6f}")


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(x_var * y_var)
    if denom == 0:
        return float("nan")
    return numerator / denom


def _rank_values(values: Sequence[float]) -> list[float]:
    sorted_values = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end][0] == sorted_values[start][0]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        for _value, index in sorted_values[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _safe_run_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    return safe or DEFAULT_RUN_ID


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _offline_fixture_csv() -> str:
    return "\n".join(
        [
            "Antigen,heavy,light,Round,Chain mutated,Design,fitness",
            "fixture,QVQLVESG,DIQMTQSP,alpha,WT,WT,0.50",
            "fixture,QVHLVESG,DIQMTQSP,alpha,HC,Q3H,0.71",
            "fixture,QVQLVESG,DIHMTQSP,alpha,LC,Q3H,0.66",
            "fixture,QVHLVESG,DIHMTQSP,beta,HC + LC,Q3H/Q3H,0.91",
            "fixture,QVQLVESG,DIQMTYSP,beta,LC,Q6Y,0.43",
            "fixture,QVKLVEYG,DIQMTQSP,beta,HC,Q3K/S7Y,0.57",
        ]
    )
