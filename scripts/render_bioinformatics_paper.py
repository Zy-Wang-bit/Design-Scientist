#!/usr/bin/env python3
"""Render the 1E62 pH-Switch Graph manuscript with the OUP/Bioinformatics template.

This script is intentionally artifact-oriented: it consumes the manuscript
bundle written by ``design-scientist generate-algorithm-paper`` and produces a
submission-style TeX/PDF package with bounded figure sizes. It does not change
the scientific content of the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BIOINFORMATICS_ABSTRACT_LIMIT = 150

SUBMISSION_PLACEHOLDER_PHRASES = (
    "to be inserted before submission",
    "to be confirmed before submission",
    "pending author confirmation",
    "not yet selected",
    "withheld for review",
    "correspondence details withheld",
    "no archival DOI is claimed",
)

STRUCTURED_ABSTRACT_HEADINGS = (
    "Motivation:",
    "Results:",
    "Availability and Implementation:",
    "Contact:",
    "Supplementary information:",
)


FIGURE_SPECS = {
    "figures/figure_1_ph_switch_graph_workflow_hermes.png": {
        "trimmed": "figures/figure_1_ph_switch_graph_workflow_hermes_trimmed.png",
        "environment": "figure*",
        "width": "0.86\\textwidth",
        "pre_caption_vspace": "1pt",
        "post_caption_vspace": "-2pt",
        "caption": (
            "pH-Switch Graph Search workflow. The model converts standardized "
            "1E62 startup data into a protonation-prior site graph, generates "
            "single-, pair-, and triplet-edit programs, scores candidates, and "
            "selects a cost-constrained computational panel. The counterfactual "
            "site map is an abstract schematic sequence-context prior, not a "
            "predicted structure, antigen-contact map, or solved 1E62-HBsAg interface."
        ),
        "alt_text": (
            "Workflow diagram from standardized 1E62 startup data to site graph, "
            "candidate generation, scoring, panel selection, and evidence-boundary outputs."
        ),
    },
    "figures/figure_2_ph_switch_graph_benchmark.png": {
        "trimmed": "figures/figure_2_ph_switch_graph_benchmark_trimmed.png",
        "environment": "figure",
        "width": "0.98\\columnwidth",
        "pre_caption_vspace": "-2pt",
        "post_caption_vspace": "-2pt",
        "caption": (
            "Synthetic benchmark comparison. Bars show mean best-in-batch "
            "selected synthetic-oracle utility with 95% confidence intervals "
            "across the fixed 6-world x 10-seed computational stress grid, not "
            "independent biological replicates. Pairwise win/tie/loss "
            "statistics against the same-pool reference are reported in Table S3."
        ),
        "alt_text": (
            "Bar chart comparing selected computational utility for pH-Switch Graph "
            "and baseline mechanisms in the stress grid."
        ),
    },
    "figures/figure_3_ph_switch_graph_project_panel.png": {
        "trimmed": "figures/figure_3_ph_switch_graph_project_panel_trimmed.png",
        "environment": "figure",
        "width": "0.98\\columnwidth",
        "pre_caption_vspace": "-2pt",
        "post_caption_vspace": "-2pt",
        "caption": (
            "Selected project candidate computational acquisition scores. Labels "
            "use standard mutation notation; candidates are computational "
            "hypotheses, not wet-lab observations."
        ),
        "alt_text": (
            "Selected 1E62 computational panel with mutation labels and acquisition "
            "scores for candidate hypotheses."
        ),
    },
    "figures/figure_4_ph_switch_graph_ablation.png": {
        "trimmed": "figures/figure_4_ph_switch_graph_ablation_trimmed.png",
        "environment": "figure",
        "width": "0.98\\columnwidth",
        "pre_caption_vspace": "-2pt",
        "post_caption_vspace": "-2pt",
        "caption": (
            "Component ablation losses. Bars show generated new-site candidate "
            "losses relative to the full mechanism; this supports candidate-space "
            "expansion, not experimental activity."
        ),
        "alt_text": (
            "Ablation bar chart showing generated new-site losses after removing "
            "pH-Switch Graph components."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_dir", type=Path, help="Directory containing manuscript.md")
    parser.add_argument("--no-compile", action="store_true", help="Only write TeX files")
    parser.add_argument(
        "--source-zip",
        type=Path,
        help="Optional path for a reproducibility source package.",
    )
    args = parser.parse_args()

    paper_dir = args.paper_dir.expanduser().resolve()
    manuscript = paper_dir / "manuscript.md"
    if not manuscript.is_file():
        raise SystemExit(f"missing manuscript: {manuscript}")
    if not (paper_dir / "references.bib").is_file():
        raise SystemExit(f"missing references.bib in {paper_dir}")

    _write_trimmed_figures(paper_dir)
    _write_hermes_image_provenance(paper_dir)
    _write_bioinformatics_bib(paper_dir)
    body_md = _body_markdown(manuscript.read_text(encoding="utf-8"))
    body_md = _remove_wide_code_blocks(body_md)
    body_md = _replace_markdown_figures(body_md)
    body_tex = _pandoc_to_latex(body_md)
    body_tex = _compact_latex_spacing(body_tex)
    body_tex = _wrap_raw_urls(body_tex)

    (paper_dir / "bioinformatics_body.md").write_text(body_md.rstrip() + "\n", encoding="utf-8")
    (paper_dir / "bioinformatics_body.tex").write_text(body_tex.rstrip() + "\n", encoding="utf-8")

    abstract = _abstract_text(manuscript.read_text(encoding="utf-8"))
    preamble = _bioinformatics_preamble(abstract, paper_dir)
    (paper_dir / "bioinformatics_preamble.tex").write_text(preamble, encoding="utf-8")
    manuscript_tex = preamble + "\n" + body_tex.rstrip() + "\n\n" + _bibliography_block() + "\n\\end{document}\n"
    (paper_dir / "bioinformatics_manuscript.tex").write_text(manuscript_tex, encoding="utf-8")

    if not args.no_compile:
        _patch_oup_template(paper_dir)
        _compile_tex(paper_dir)
    if args.source_zip:
        _write_source_package(paper_dir, args.source_zip.expanduser().resolve())
    return 0


def _write_trimmed_figures(paper_dir: Path) -> None:
    try:
        from PIL import Image, ImageChops
    except Exception as exc:  # pragma: no cover - depends on local optional package
        if not os.environ.get("DS_RENDER_BIOINFO_REEXEC") and shutil.which("uv"):
            env = dict(os.environ)
            env["DS_RENDER_BIOINFO_REEXEC"] = "1"
            os.execvpe("uv", ["uv", "run", "python", __file__, *sys.argv[1:]], env)
        raise SystemExit("Pillow is required to trim Bioinformatics figures") from exc

    for source_name, spec in FIGURE_SPECS.items():
        source = paper_dir / source_name
        target = paper_dir / str(spec["trimmed"])
        if not source.is_file():
            continue
        if target.is_file():
            target.unlink()
        image = Image.open(source).convert("RGB")
        background = Image.new("RGB", image.size, (255, 255, 255))
        diff = ImageChops.difference(image, background).convert("L")
        mask = diff.point(lambda pixel: 255 if pixel > 8 else 0)
        bbox = mask.getbbox()
        if bbox is None:
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            continue
        left, top, right, bottom = bbox
        horizontal_margin = 16
        top_margin = 16
        bottom_margin = 0
        left = max(0, left - horizontal_margin)
        top = max(0, top - top_margin)
        right = min(image.width, right + horizontal_margin)
        bottom = min(image.height, bottom + bottom_margin)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(target)


def _write_hermes_image_provenance(paper_dir: Path) -> None:
    figure = paper_dir / "figures" / "figure_1_ph_switch_graph_workflow_hermes.png"
    if not figure.is_file():
        return
    digest = _sha256(figure)
    matches = []
    hermes_cache = Path.home() / ".hermes" / "cache" / "images"
    if hermes_cache.is_dir():
        for cache_file in sorted(hermes_cache.glob("*gpt-image-2*.png")):
            if cache_file.is_file() and _sha256(cache_file) == digest:
                matches.append(str(cache_file))
    payload = {
        "schema_version": 1,
        "generator": "Hermes",
        "model": _infer_hermes_model(matches) or "gpt-image-2",
        "paper_figure": str(figure),
        "sha256": digest,
        "matching_hermes_cache_files": matches,
        "provenance_status": "matched_hermes_cache" if matches else "figure_present_cache_match_not_found",
    }
    (paper_dir / "figures" / "figure_1_ph_switch_graph_workflow_hermes_provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _infer_hermes_model(matches: list[str]) -> str | None:
    if not matches:
        return None
    name = Path(matches[0]).name
    match = re.search(r"_(gpt-image-2(?:-[A-Za-z0-9]+)?)_", name)
    return match.group(1) if match else None


def _body_markdown(text: str) -> str:
    lines = text.splitlines()
    start = None
    end = len(lines)
    for index, line in enumerate(lines):
        if line.strip() == "## Introduction":
            start = index
        elif start is not None and line.strip() == "## References":
            end = index
            break
    if start is None:
        raise SystemExit("manuscript.md does not contain a ## Introduction section")
    return "\n".join(lines[start:end]).strip() + "\n"


def _abstract_text(text: str) -> str:
    match = re.search(r"^## Abstract\s+(.+?)(?=^## )", text, flags=re.M | re.S)
    if not match:
        raise SystemExit("manuscript.md does not contain a ## Abstract section")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", match.group(1)) if part.strip()]
    return paragraphs[0] if paragraphs else ""


def _replace_markdown_figures(text: str) -> str:
    for source_name, spec in FIGURE_SPECS.items():
        pattern = re.compile(rf"^!\[[^\]]*\]\({re.escape(source_name)}\)\s*$", flags=re.M)
        figure = _latex_figure(
            path=str(spec["trimmed"]),
            environment=str(spec["environment"]),
            width=str(spec["width"]),
            caption=str(spec["caption"]),
            alt_text=str(spec.get("alt_text", "")),
            pre_caption_vspace=str(spec.get("pre_caption_vspace", "-2pt")),
            post_caption_vspace=str(spec.get("post_caption_vspace", "-2pt")),
        )
        text = pattern.sub(lambda _match: figure, text)
    return text


def _remove_wide_code_blocks(text: str) -> str:
    """Drop wide lifecycle code blocks that do not fit the two-column template."""

    return re.sub(
        r"\n```text\nfit_state\(observed sequences, endpoints\).*?select_panel\(scores, cost_budget\).*?```\n",
        "\n",
        text,
        flags=re.S,
    )


def _latex_figure(
    *,
    path: str,
    environment: str,
    width: str,
    caption: str,
    alt_text: str,
    pre_caption_vspace: str,
    post_caption_vspace: str,
) -> str:
    return "\n".join(
        [
            f"\\begin{{{environment}}}[!t]",
            "\\centering",
            f"\\includegraphics[width={width},keepaspectratio]{{{path}}}",
            f"\\vspace{{{pre_caption_vspace}}}",
            f"\\caption{{{_escape_latex(caption)}}}",
            f"\\noindent\\textbf{{Alt text:}} {_escape_latex(alt_text)}",
            f"\\vspace{{{post_caption_vspace}}}",
            f"\\end{{{environment}}}",
        ]
    )


def _pandoc_to_latex(markdown: str) -> str:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise SystemExit("pandoc is required to render the Bioinformatics TeX body")
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "body.md"
        output_path = Path(tmp) / "body.tex"
        input_path.write_text(markdown, encoding="utf-8")
        subprocess.run(
            [
                pandoc,
                "--from=markdown+citations+raw_tex",
                "--to=latex",
                "--natbib",
                "--wrap=none",
                "--syntax-highlighting=none",
                "--shift-heading-level-by=-1",
                str(input_path),
                "-o",
                str(output_path),
            ],
            check=True,
        )
        return output_path.read_text(encoding="utf-8")


def _compact_latex_spacing(text: str) -> str:
    text = text.replace("\n\n\\begin{figure}", "\n\\begin{figure}")
    text = text.replace("\n\n\\begin{figure*}", "\n\\begin{figure*}")
    text = text.replace("\\end{figure}\n\n", "\\end{figure}\n")
    text = text.replace("\\end{figure*}\n\n", "\\end{figure*}\n")
    return text


def _wrap_raw_urls(text: str) -> str:
    """Wrap raw prose URLs so two-column TeX can break them safely."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        url = raw.rstrip(".,;")
        return f"\\url{{{url}}}{raw[len(url):]}"

    return re.sub(r"(?<![{])https?://[^\s}]+", replace, text)


def _bioinformatics_preamble(abstract: str, paper_dir: Path) -> str:
    del abstract
    structured_abstract = _structured_bioinformatics_abstract(paper_dir)
    _validate_structured_bioinformatics_abstract(structured_abstract)
    escaped_abstract = _escape_latex(structured_abstract)
    author = _escape_latex(os.environ.get("DS_PAPER_AUTHORS", "Author details to be inserted before submission"))
    affiliation = _escape_latex(
        os.environ.get("DS_PAPER_AFFILIATION", "Affiliation details to be inserted before submission")
    )
    contact = _escape_latex(os.environ.get("DS_PAPER_CONTACT", "Corresponding author email to be inserted before submission"))
    return "\n".join(
        [
            "\\documentclass[unnumsec,webpdf,contemporary,large]{oup-authoring-template}",
            "\\graphicspath{{figures/}}",
            "\\usepackage{graphicx}",
            "\\usepackage{booktabs}",
            "\\usepackage{array}",
            "\\usepackage{url}",
            "\\usepackage{xurl}",
            "\\usepackage{hyperref}",
            "\\usepackage{microtype}",
            "\\hypersetup{breaklinks=true}",
            "\\Urlmuskip=0mu plus 1mu",
            "\\setlength{\\abovecaptionskip}{0pt}",
            "\\setlength{\\belowcaptionskip}{-2pt}",
            "\\setlength{\\floatsep}{2pt plus 1pt minus 1pt}",
            "\\setlength{\\textfloatsep}{2pt plus 1pt minus 1pt}",
            "\\setlength{\\intextsep}{2pt plus 1pt minus 1pt}",
            "\\setlength{\\dblfloatsep}{2pt plus 1pt minus 1pt}",
            "\\setlength{\\dbltextfloatsep}{2pt plus 1pt minus 1pt}",
            "\\raggedbottom",
            "\\hfuzz=12pt",
            "\\begin{document}",
            "\\journaltitle{Bioinformatics}",
            "\\appnotes{Methods}",
            "\\firstpage{1}",
            "\\copyrightyear{2026}",
            "\\pubyear{2026}",
            "\\title[pH-Switch Graph Search]{pH-Switch Graph Search for Auditable New-Site Hypothesis Generation in Sparse Antibody pH-Switch Design}",
            f"\\author[1,$\\ast$]{{{author}}}",
            f"\\address[1]{{{affiliation}}}",
            f"\\corresp[$\\ast$]{{{contact}}}",
            f"\\abstract{{{escaped_abstract}}}",
            (
                "\\keywords{antibody design, pH-switch engineering, generative design, "
                "active learning, small-data protein engineering}"
            ),
            "\\maketitle",
            "",
        ]
    )


def _structured_bioinformatics_abstract(paper_dir: Path) -> str:
    repository_url = os.environ.get(
        "DS_PAPER_REPOSITORY_URL",
        "https://github.com/Zy-Wang-bit/Design-Scientist",
    )
    archive_url = os.environ.get(
        "DS_PAPER_ARCHIVE_URL",
        "archival DOI or Software Heritage URL to be inserted before submission",
    )
    contact = os.environ.get("DS_PAPER_CONTACT", "corresponding author email to be inserted before submission")
    external_ph_clause = _external_ph_switch_abstract_clause(paper_dir)
    return (
        "Motivation: Sparse antibody pH-switch campaigns need algorithms that propose "
        "testable mutation sites beyond measured candidates. "
        "Results: pH-Switch Graph Search builds a protonation-prior site graph, generates "
        "one- to three-edit heavy/light-chain hypotheses, and selects a cost-constrained 1E62 panel. "
        "Computational stress tests and public pH-switch replays provide bounded, auditable checks; "
        f"{external_ph_clause} "
        "Prospective wet-lab activity is left for validation. "
        f"Availability and Implementation: Code and test artifacts are available at {repository_url}; "
        f"the submission version is archived at {archive_url}. "
        f"Contact: {contact}. "
        "Supplementary information: Supplementary Data are available with the manuscript."
    )


def _external_ph_switch_abstract_clause(paper_dir: Path) -> str:
    summary = paper_dir / "tables" / "external_ph_switch_benchmark_summary.csv"
    if not summary.is_file():
        return "public pH-switch replay data are indexed in the supplement."
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    transfer = _summary_row(rows, "leave_one_study_transition_calibration")
    histidine = _summary_row(rows, "histidine_count")
    if not transfer:
        return "public pH-switch replay data are indexed in the supplement."
    dataset_count = _compact_number(transfer.get("dataset_count"))
    transfer_norm = _compact_number(transfer.get("mean_best_selected_normalized_log_ratio"))
    histidine_norm = _compact_number(histidine.get("mean_best_selected_normalized_log_ratio")) if histidine else "n/a"
    variant_clause = _external_ph_switch_variant_abstract_clause(paper_dir)
    return (
        f"a {dataset_count}-table public pH-switch replay reached normalized log-ratio "
        f"{transfer_norm} versus {histidine_norm} for histidine-count selection{variant_clause}."
    )


def _external_ph_switch_variant_abstract_clause(paper_dir: Path) -> str:
    summary_path = paper_dir / "algorithm_results_summary.json"
    if not summary_path.is_file():
        return ""
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    benchmark = payload.get("external_ph_switch_benchmark")
    if not isinstance(benchmark, dict):
        return ""
    variant_summary = benchmark.get("variant_replay_summary")
    if not isinstance(variant_summary, dict) or not variant_summary:
        return ""
    variant_count = _compact_number(variant_summary.get("variant_count"))
    correlation = _compact_number(
        variant_summary.get("predicted_vs_observed_normalized_log_ratio_pearson")
    )
    context_enrichment = ""
    for row in variant_summary.get("method_summaries", []):
        if isinstance(row, dict) and row.get("method") == "transition_context_prior":
            context_enrichment = _compact_number(
                row.get("top_tertile_enrichment_at_top_third_by_score")
            )
            break
    if not variant_count or not correlation:
        return ""
    suffix = f"; leave-one-variant replay over {variant_count} variants gave r={correlation}"
    if context_enrichment:
        suffix += f" and context-prior enrichment {context_enrichment}"
    return suffix


def _summary_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    for row in rows:
        if row.get("method") == method and row.get("dataset_id") == "overall":
            return row
    return {}


def _compact_number(value: object) -> str:
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return str(value or "")
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _validate_structured_bioinformatics_abstract(text: str) -> None:
    plain = _plain_abstract_text(text)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.:-]*", plain)
    if len(words) > BIOINFORMATICS_ABSTRACT_LIMIT:
        raise SystemExit(
            f"Bioinformatics abstract has {len(words)} words; limit is {BIOINFORMATICS_ABSTRACT_LIMIT}"
        )
    missing = [heading for heading in STRUCTURED_ABSTRACT_HEADINGS if heading not in plain]
    if missing:
        raise SystemExit("Bioinformatics abstract missing required heading(s): " + ", ".join(missing))
    lower = plain.lower()
    placeholders = [phrase for phrase in SUBMISSION_PLACEHOLDER_PHRASES if phrase.lower() in lower]
    if placeholders and os.environ.get("DS_ALLOW_SUBMISSION_PLACEHOLDERS") != "1":
        raise SystemExit(
            "Bioinformatics abstract contains unresolved submission placeholder(s): "
            + ", ".join(placeholders)
        )


def _plain_abstract_text(text: str) -> str:
    plain = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    plain = plain.replace("\\\\", " ")
    plain = re.sub(r"\\[A-Za-z]+\{?", " ", plain)
    plain = plain.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", plain).strip()


def _bibliography_block() -> str:
    return "\n".join(
        [
            "\\bibliographystyle{oup-abbrvnat}",
            "\\bibliography{references_bioinformatics}",
        ]
    )


def _write_bioinformatics_bib(paper_dir: Path) -> None:
    source = paper_dir / "references.bib"
    target = paper_dir / "references_bioinformatics.bib"
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if re.match(r"\s*note\s*=", line):
            continue
        lines.append(line)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _patch_oup_template(paper_dir: Path) -> None:
    cls_path = paper_dir / "oup-authoring-template.cls"
    if not cls_path.is_file():
        return
    text = cls_path.read_text(encoding="utf-8", errors="replace")
    patched = text
    for snippet in (
        r"\hfill{\color{black!20}\rule{45pt}{55pt}}",
        r"\hfill\hbox{\color{black!20}\rule[-7.5pt]{55pt}{20pt}}",
        r"\hfill\hbox{\color{black!20}\rule{55pt}{20pt}}",
        r"\hfill\hbox{\color{black!20}\rule{55pt}{20pt}}\hskip4mm",
    ):
        patched = patched.replace(snippet, "")
    for header in (
        r"\@journaltitle, \@pubyear,\ Volume \@vol,\ Issue \@issue",
        r"\rightmark, Volume \@vol, Issue \@issue",
    ):
        patched = patched.replace(header, header.split(",")[0])
    if patched != text:
        cls_path.write_text(patched, encoding="utf-8")


def _compile_tex(paper_dir: Path) -> None:
    compiler = shutil.which("tectonic")
    if not compiler:
        raise SystemExit("tectonic is required to compile bioinformatics_manuscript.tex")
    subprocess.run(
        [
            compiler,
            "--keep-logs",
            "--keep-intermediates",
            "bioinformatics_manuscript.tex",
        ],
        cwd=paper_dir,
        check=True,
    )


def _write_source_package(paper_dir: Path, output_zip: Path) -> None:
    repo_root = _find_repo_root(paper_dir)
    run_dir = paper_dir.parent
    project_dir = run_dir.parent.parent
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    file_paths: set[Path] = set()
    for relative in (
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "scripts/render_bioinformatics_paper.py",
        "projects/1e62_ph_sensitive_v3_start/runs/ph_switch_graph_1e62_research/paper/oup-authoring-template.cls",
        "src/design_scientist/algorithms/pcig.py",
        "src/design_scientist/sequence_annotation.py",
        "src/design_scientist/generative_benchmark.py",
        "src/design_scientist/external_antibody_benchmark.py",
        "src/design_scientist/project_replay.py",
        "src/design_scientist/manuscript.py",
        "src/design_scientist/cli.py",
        "tests/test_pcig.py",
        "tests/test_sequence_annotation.py",
        "tests/test_external_antibody_benchmark.py",
        "tests/test_manuscript.py",
        "tests/test_generative_benchmark.py",
        "tests/test_project_replay_cmdgd.py",
    ):
        path = repo_root / relative
        if path.is_file():
            file_paths.add(path)

    for directory in (
        paper_dir / "figures",
        paper_dir / "tables",
        run_dir / "mechanism_nodes" / "ph_switch_graph",
        project_dir / "standardized",
        project_dir / "framework",
    ):
        if directory.is_dir():
            file_paths.update(path for path in directory.rglob("*") if path.is_file())

    for pattern in ("*.csv", "*.json", "*.md"):
        file_paths.update(path for path in run_dir.glob(pattern) if path.is_file())

    for relative in (
        "manuscript.md",
        "bioinformatics_manuscript.tex",
        "bioinformatics_manuscript.pdf",
        "bioinformatics_preamble.tex",
        "bioinformatics_body.md",
        "bioinformatics_body.tex",
        "references.bib",
        "references_bioinformatics.bib",
        "bioinformatics_manuscript.bbl",
        "supplement_manifest.md",
        "reproducibility_contract.md",
        "oracle_and_stress_worlds.md",
        "algorithm_formal_definition.md",
        "data_dictionary.json",
        "algorithm_hyperparameters.json",
        "developability_risk_summary.json",
        "claim_boundary_analysis.json",
        "data_availability.md",
        "code_availability.md",
        "submission_metadata.md",
        "figure_alt_text.json",
        "supplementary_data.md",
        "submission_package_manifest.md",
        "paper_readiness_report.json",
        "independent_reviewer_summary.md",
    ):
        path = paper_dir / relative
        if path.is_file():
            file_paths.add(path)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(file_paths):
            archive.write(path, path.relative_to(repo_root))


def _find_repo_root(path: Path) -> Path:
    current = path.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise SystemExit(f"could not find repository root from {path}")


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


if __name__ == "__main__":
    raise SystemExit(main())
