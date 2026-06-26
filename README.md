# SKINET — Automated Kidney-Biopsy Instance Segmentation

Two-stage instance segmentation and morphometry for Masson's trichrome kidney-biopsy
whole-slide images (WSIs): a **Mask2Former** (Swin-B) detector localizes and segments
ten cortical renal structure classes, and a promptable **SAM-Med2D** refiner with a
lightweight, trainable **Kidney Adapter** cleans up instance boundaries from the
detector's boxes. A command-line tool (`skinet`) submits the whole pipeline to an
SGE/HPC cluster and reports detection, Dice, and F1 metrics.

> Research code accompanying the paper *“Transformer-Based Instance Segmentation
> with an Adapter-Tuned SAM-Med2D Refiner for Automated Kidney-Biopsy
> Morphometry.”* Developed at the LEAD Laboratory (CNRS UMR 5022, Université
> Bourgogne Europe) and CHU Dijon Bourgogne.

---

## Pipeline

```
WSI ─► tile (1024, stride 683) ─► Mask2Former (Swin-B) ─► stitch to global coords
    ─► fuse (same/cross class) ─► hierarchy reconciliation (cleanup/priority/orphan)
    ─► SAM-Med2D + Kidney Adapter refine  ─► off-tissue guard ─► instances + grades
```

A second low-magnification model segments cortex / capsule / medulla to guide
region-of-interest selection and normalize morphometry.

## Install

The CLI is a standard Python package:

```bash
git clone https://github.com/<you>/skinet.git
cd skinet
pip install --user .          # installs the `skinet` command
```

The CLI uses only the standard library. The heavy dependencies (PyTorch,
detectron2, SAM-Med2D) live in a separate conda environment used by the compute
job — see [`docs/installation.md`](docs/installation.md) and
[`environment.yml`](environment.yml).

## Configure

Paths to weights, data, and the conda environment are read from a small config
(env vars or a `skinet.env` file). Copy the example and check it:

```bash
cp skinet/skinet.env.example skinet.env
skinet where      # show resolved paths
skinet doctor     # verify everything is reachable
```

The pipeline scripts read their roots from `SKINET_ROOT` / `SKINET_OLD_ROOT`
(see `skinet/scripts/e7_config.py`), so the code is portable across installs.

## Use

```bash
skinet run  <slide>          # segment one slide -> overlay + masks
skinet test <SOURCE_KEY>     # segment one labelled source + metrics
skinet all                   # whole test split + metrics
skinet eval                  # (re)compute metrics on an existing output
skinet bench                 # timing benchmark (no metrics)
skinet status | watch <ID>   # queue / live log
```

Add `--sam` for SAM-Med2D refinement; `--sam-mode replace` runs the
detector-as-box-only variant (SAM as sole segmenter).

Metrics (per-class P/R/F1, instance Dice, semantic Dice, confusion matrix)
require ground-truth annotations, so they are produced for the labelled test
split (`test` / `all` / `eval`), not for an arbitrary `run <slide>`.

## Run without a scheduler (raw Python / shell)

To run directly on a server (no qsub), use the wrapper, which sets up the
environment and calls `scripts/e7_run.py` in the foreground:

```bash
bash run_local.sh /path/to/slide.jpg            # -> overlay + masks
bash run_local.sh /path/to/slide.jpg --sam      # + SAM-Med2D refinement
```

Or call the pipeline script yourself after activating the environment:

```bash
conda activate skinet
export PYTHONPATH=/path/to/Mask2Former:/path/to/SAM-Med2D
python skinet/scripts/e7_run.py \
    --image /path/to/slide.jpg --source slide \
    --weights <model_best.pth> --split-csv <split.csv> \
    --scripts-dir skinet/scripts --out-dir ./out --overlay
```

Paths are configurable via environment variables (`SKINET_ROOT`,
`SKINET_WEIGHTS`, `SKINET_CONDA_SH`, ...); see the top of `run_local.sh`.
This needs a GPU-visible machine.

## Repository layout

```
skinet/
  cli.py                 the `skinet` command (submits to the cluster)
  jobs/submit_e7_run.sh  SGE job script (runs the pipeline + metrics)
  skinet.env.example     configuration template
  scripts/               pipeline stages (prep, train, infer, refine, eval)
docs/                    installation + usage notes
environment.yml          conda environment for the compute side
pyproject.toml           package metadata
```

## Data and weights

Patient data and trained weights are **not** included (clinical data; large
binaries). The `.gitignore` blocks `*.pth`, `*.jp2`, `*.xml`, `outputs/`, etc.
Point the config at your own copies.

## Citation

```bibtex
@article{khowarizmi_skinet,
  title  = {Transformer-Based Instance Segmentation with an Adapter-Tuned
            SAM-Med2D Refiner for Automated Kidney-Biopsy Morphometry},
  author = {Khowarizmi, Muhammad Haekal and Paindavoine, Michel and
            Bard, Patrick and Legendre, Mathieu},
  year   = {2026}
}
```

## License

Not yet licensed. Until a license is added, default copyright applies (all
rights reserved). Contact the authors before reuse.
