# Project working notes — RMT pruning paper restructure

Maintenance notes for the in-progress paper restructure of
**"Pruning Deep Neural Networks via the Marchenko–Pastur Distribution"**
by Berlyand, Bourdais, Owhadi, Shmalo. Repository:
<https://github.com/yspennstate/penstate/RMT_based_pruning_in_deep_learning>.

---

## State of play (2026-05-09)

There are now **two papers** that compile from this repo:

| Paper | TeX | PDF | Pages | What it is |
|---|---|---|---|---|
| Main manuscript | `paper/main.tex` | `paper/main.pdf` | **60** | Theory + headline numerical results. Must keep all theory + Tab 2 + new FLOP table + theory↔numerics map + speedup discussion. |
| Methodology companion | `cast_2e/methodology.tex` | `cast_2e/methodology.pdf` | **54** | Full methodology + extended numerics + migrated appendices + comprehensive code-to-numbers map. |
| Original (archival) | `paper/main_original_92pages.tex` | — | 92 (was) | The pre-split source preserved for diff/audit. Do **not** modify; treat as read-only reference. |

Both PDFs build clean: 0 undefined refs, 0 undefined cites, 0 overfull
hbox warnings (margins clean).

### Migration map (already done)

The following content was moved from main paper into methodology paper
**verbatim**, in the order shown:

| Original main location | Original lines | Methodology section | Methodology label |
|---|---|---|---|
| App B (Other Numerical Results) | 2333–2528 | §6 | `sec:fc_num_meth` |
| App E (BEMA / RMT diagnostics) | 3888–4095 | §7 | `sec:bema` |
| App F (Spectral Edge Budgeting + hybrid protocols) | 4096–4508 | §8 | `sec:seb-protocols` |
| App G (CAST 2:4 + ToMe original) | 4509–4612 | §9 | `sec:cast-old-app` |
| §3.3 (Interpretation) | 298–302 | §13 | `sec:s33-interpretation` |
| §3.4 (Broader architecture validation) | 303–361 | §14 | `sec:s34-arch-validation` |
| §3.5 (Theory↔numerics) | 362–522 | §15 | `sec:s35-cert-numerics` |
| §3.6 (vs prior pruning on ViT-B/16) | 523–568 | §16 | `sec:s36-vitb-comparison` |
| §3.7 (vs prior pruning on DeiT) | 569–617 | §17 | `sec:s37-deit-comparison` |

The methodology paper additionally has new content (§3–§5 for CAST-2E
methodology, §10 for the cert audit on 282 cells × 18 archs, §11 for
detailed §3 numerics, §12 for the quick-reference index from main-paper
topics to methodology sections).

### Cross-paper reference commands
- In main: `\methref{X}` renders as "the methodology paper" (X is a
  topic key listed in the methodology §12 index).
- In methodology: `\mainref{X}` renders as "the main paper".

---

## Repository layout

```
RMT_based_pruning_in_deep_learning/
├── README.md                        — top-level repo README
├── paper/                           — main manuscript
│   ├── main.tex / main.pdf          — current 60-page main
│   ├── main_original_92pages.tex    — archival pre-split source
│   ├── rmt_vit_pruning.bib          — shared bib (95 entries)
│   └── *.pdf, *.png, *.jpg          — figures
├── cast_2e/                         — methodology companion
│   ├── methodology.tex / .pdf       — current 54-page companion
│   ├── README.md                    — CAST-2E subsystem overview (25 .py files documented)
│   ├── AUDIT.md                     — full data audit
│   ├── THEORY.md                    — theory↔code map
│   ├── PERMUTATION_ALIGN_DESIGN.md  — Cin perm design doc
│   ├── SIMULATION_PLAN.md           — sweep plan
│   ├── REPORT.md                    — run report
│   ├── rmt_vit_pruning.bib          — synced copy
│   ├── code/                        — 25 .py files (project_kn_sparsity.py, run_*_ft_inline.py,
│   │                                  benchmark_*.py, mac_counter.py, cert_opt_eval_*.py)
│   ├── scripts/                     — 16 pod-side launch scripts
│   ├── benchmarks/                  — A100/L4 throughput JSONs
│   ├── benchmarks_speedup/          — 6:12 → 2:4 projection-speedup JSONs
│   ├── sweep_results/               — k:n cell-sweep JSONs (Pod 3a)
│   ├── sweep_results_initial/       — first-round sweep JSONs
│   ├── post_ft_eval/                — post-FT eval JSONs
│   ├── masks_archive/               — per-layer mask stats
│   └── *.png/jpg/pdf                — figures referenced by methodology paper
├── adaptive_rmt/                    — RMT diagnostics package (existing)
├── configs/                         — sweep configs
├── model_queue_runs/                — cached SVD/ESD per model
├── scripts/                         — repo-level launchers
├── src/                             — inherited src/ (RMT/SplittableLayers/training/utils)
└── *.py                             — ~60 root scripts (hybrid_mag*, haar_*, iterative_*, etc.)
```

---

## Current maintenance tasks

The current iteration goal is to make the methodology paper as comprehensive
as possible. Recurring review tasks:
- Identify content that can move from the main manuscript to the methodology paper.
- Check whether any material from the original 92-page paper remains unmigrated.
- Audit references, citations, figures, and margins.
- Confirm that the methodology paper covers every implemented method.
- Identify relevant pruning works that should be cited.

For structural changes, show diffs before applying or pushing them. Do not push
changes without explicit approval.

---

## Useful commands

```bash
# Build main paper (run from paper/)
cd paper && pdflatex -interaction=batchmode main.tex && bibtex main && \
  pdflatex -interaction=batchmode main.tex && pdflatex -interaction=batchmode main.tex

# Build methodology (run from cast_2e/)
cd cast_2e && pdflatex -interaction=batchmode methodology.tex && \
  bibtex methodology && pdflatex -interaction=batchmode methodology.tex && \
  pdflatex -interaction=batchmode methodology.tex

# Audit refs/cites/margins after build
grep "Reference \`" paper/main.log | grep -oE "Reference \`[^']+'" | sort -u
grep "Citation \`" paper/main.log | grep -oE "Citation \`[^']+'" | sort -u
grep -c "Overfull \\\\hbox" paper/main.log

# Compare current main vs original
diff <(sed -n '1,300p' paper/main_original_92pages.tex) <(sed -n '1,300p' paper/main.tex)
```

## Hard rules

- **Theory remains in the main paper.** Sections §1–§5 of the body and
  Appendices A (Gaussian specialization), C (perturbation lemma proofs),
  D (low-rank deformations / main proofs) **must not be moved** to the
  methodology paper. They are the mathematical contribution.
- **Tab 2 (multi-arch Hybrid Mag–SER) and the new FLOP table** stay in
  main. They are the headline empirical results.
- The repository at <https://github.com/yspennstate/RMT_based_pruning_in_deep_learning>
  is public; do not commit secrets or large checkpoints (`*.pt` is in
  `.gitignore`). The 30 GB of post-FT checkpoints will go to HuggingFace
  Hub when the paper is finalised, not to git.
- LaTeX build artifacts (`*.aux`, `*.out`, `*.toc`, `*.bbl`, `*.blg`,
  `*.fls`, `*.fdb_latexmk`, `*.lof`, `*.lot`, `*.log`) are in `.gitignore`.
