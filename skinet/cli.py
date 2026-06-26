#!/usr/bin/env python3
"""
skinet -- command-line launcher for the SKINET kidney-biopsy WSI pipeline.

Installed via `pip install .`, this exposes a `skinet` command that submits jobs
to the CCuB SGE cluster (qsub) behind a simple interface:

    skinet run  <slide.jp2>        segment ONE slide you point at
    skinet test <SOURCE_KEY>       segment ONE source from the test split
    skinet all                     segment the whole test split
    skinet bench                   whole test split, tissue filter OFF (timing)
    skinet status                  your jobs in the queue
    skinet watch <JOB_ID>          follow a job's log live
    skinet where                   print resolved paths
    skinet doctor                  check your setup before submitting

Extra pipeline flags pass straight through, e.g.:
    skinet all --sam --sam-mode replace
    skinet run x.jp2 --tissue-debug

Configuration is read (in order) from: $SKINET_ENV, ./skinet.env,
~/.config/skinet/skinet.env, then built-in defaults; any KEY can also be
overridden by an environment variable of the same name. See `skinet where`.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from importlib.resources import files as _res_files  # py3.9+
except Exception:  # pragma: no cover
    _res_files = None


# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #
DEFAULTS = {
    # (A) shared, read-only, called by absolute path
    "SKINET_SCRIPTS_DIR": "",   # "" => auto: use packaged scripts if vendored,
                                #        else fall back to the repo path below
    "SKINET_CONDA_SH":    "/work/lead/mk657917/miniconda3/etc/profile.d/conda.sh",
    "SKINET_CONDA_ENV":   "skinet",
    "SKINET_M2F_LIB":     "/work/lead/mk657917/skinet/models/Mask2Former",
    "SKINET_SAM_LIB":     "/work/lead/mk657917/skinet/models/SAM-Med2D",
    "SKINET_MODULES":     "gcc/11.2.0 cuda/11.8",
    "SKINET_WEIGHTS":     "/work/lead/mk657917/SKINET/outputs/e7_clinical_pipeline/main_m2f_swin_b/checkpoints/model_best.pth",
    "SKINET_SPLIT_CSV":   "/work/lead/mk657917/SKINET/outputs/preprocessing/e7_clinical_pipeline/02_e7_source_split.csv",
    "SKINET_SAM_BEST":    "",   # optional; resolved by the stage default if empty
    "SKINET_SAM_CKPT":    "",
    # (B) yours, writable -- per-user folder under /work (resolved at runtime).
    #     $USER expands to your login name; falls back to $HOME if /work absent.
    "SKINET_WORKDIR":     "/work/lead/$USER/skinet-runs",
    # cluster
    "SKINET_QUEUE":       "gpu",
}

CONFIG_KEYS = list(DEFAULTS.keys())

# repo fallback used only when the package does not carry a vendored script set
REPO_SCRIPTS_FALLBACK = "/work/lead/mk657917/SKINET/scripts/10_e7_clinical_pipeline"


def packaged_dir(sub: str) -> Path:
    """Path to a subfolder shipped inside the installed package."""
    if _res_files is not None:
        try:
            return Path(str(_res_files("skinet") / sub))
        except Exception:
            pass
    return Path(__file__).resolve().parent / sub


def _resolve_scripts_dir(raw: str) -> str:
    """If the user set SKINET_SCRIPTS_DIR, honour it. Otherwise prefer the
    packaged scripts (when vendor_scripts.sh has marked them complete via a
    VENDORED file), else fall back to the repo path."""
    if raw:
        return raw
    pkg = packaged_dir("scripts")
    if (pkg / "VENDORED").is_file() and (pkg / "e7_run.py").is_file():
        return str(pkg)
    return REPO_SCRIPTS_FALLBACK


def _parse_env_file(path: Path) -> dict:
    out = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k in CONFIG_KEYS:
                out[k] = v
    except Exception:
        pass
    return out


def load_config() -> dict:
    """defaults  <-  config file  <-  environment variables."""
    cfg = dict(DEFAULTS)
    candidates = []
    if os.environ.get("SKINET_ENV"):
        candidates.append(Path(os.environ["SKINET_ENV"]))
    candidates.append(Path.cwd() / "skinet.env")
    candidates.append(Path.home() / ".config" / "skinet" / "skinet.env")
    chosen = None
    for c in candidates:
        if c.is_file():
            cfg.update(_parse_env_file(c))
            chosen = c
            break
    for k in CONFIG_KEYS:                      # env vars win
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    cfg["_config_file"] = str(chosen) if chosen else "(built-in defaults)"
    cfg["SKINET_SCRIPTS_DIR"] = _resolve_scripts_dir(cfg["SKINET_SCRIPTS_DIR"])
    cfg["_workdir"] = _resolve_workdir(cfg["SKINET_WORKDIR"])
    cfg["SKINET_WORKDIR"] = str(cfg["_workdir"])
    cfg["_logdir"] = cfg["_workdir"] / "logs"
    cfg["_outdir"] = cfg["_workdir"] / "outputs"
    return cfg


def _resolve_workdir(raw: str) -> Path:
    """Expand $USER/$HOME. If the result points under /work but /work/lead does
    not exist (e.g. running off-cluster), fall back to ~/skinet-runs."""
    import getpass
    os.environ.setdefault("USER", getpass.getuser())
    wd = os.path.expandvars(raw)
    if "$" in wd:                       # a variable failed to expand
        wd = str(Path.home() / "skinet-runs")
    if wd.startswith("/work/") and not Path("/work/lead").is_dir():
        wd = str(Path.home() / "skinet-runs")
    return Path(wd)


def job_script_path() -> Path:
    """Resolve the packaged SGE job script (works after pip install)."""
    if _res_files is not None:
        try:
            p = _res_files("skinet") / "jobs" / "submit_e7_run.sh"
            return Path(str(p))
        except Exception:
            pass
    return Path(__file__).resolve().parent / "jobs" / "submit_e7_run.sh"


def write_effective_env(cfg: dict) -> Path:
    """Write the resolved config to the workdir so the compute node can source
    exactly what the CLI resolved (avoids cramming many vars, some with spaces,
    into qsub -v)."""
    cfg["_logdir"].mkdir(parents=True, exist_ok=True)
    eff = cfg["_workdir"] / ".skinet-effective.env"
    lines = ["# auto-generated by `skinet`; do not edit (regenerated each run)\n"]
    for k in CONFIG_KEYS:
        lines.append(f'{k}="{cfg[k]}"\n')
    eff.write_text("".join(lines))
    return eff


# --------------------------------------------------------------------------- #
#  pretty printing
# --------------------------------------------------------------------------- #
_TTY = sys.stdout.isatty()
def _c(code, s): return f"\033[{code}m{s}\033[0m" if _TTY else s
def say(*a):  print(_c("36;1", "skinet"), *a)
def ok(*a):   print(" ", _c("32", "OK"), *a)
def warn(*a): print(" ", _c("33", "!"), *a)
def die(msg, code=1):
    print(" ", _c("31", "x"), msg, file=sys.stderr); sys.exit(code)


# --------------------------------------------------------------------------- #
#  submission
# --------------------------------------------------------------------------- #
def _shared_pipeline_args(cfg: dict) -> list:
    a = []
    if cfg["SKINET_WEIGHTS"]:   a += ["--weights", cfg["SKINET_WEIGHTS"]]
    if cfg["SKINET_SPLIT_CSV"]: a += ["--split-csv", cfg["SKINET_SPLIT_CSV"]]
    if cfg["SKINET_SAM_BEST"]:  a += ["--sam-best", cfg["SKINET_SAM_BEST"]]
    if cfg["SKINET_SAM_CKPT"]:  a += ["--sam-checkpoint", cfg["SKINET_SAM_CKPT"]]
    return a


def preflight(cfg: dict):
    sdir = Path(cfg["SKINET_SCRIPTS_DIR"])
    if not (sdir / "e7_run.py").is_file():
        die(f"e7_run.py not found in {sdir}\n"
            f"      set SKINET_SCRIPTS_DIR (see `skinet where`)")
    if not shutil.which("qsub"):
        die("qsub not found -- are you on a CCuB login node?")
    try:
        cfg["_logdir"].mkdir(parents=True, exist_ok=True)
        cfg["_outdir"].mkdir(parents=True, exist_ok=True)
    except Exception:
        die(f"cannot create your work folder: {cfg['_workdir']}\n"
            f"      set SKINET_WORKDIR to somewhere you can write")


def submit(cfg: dict, label: str, pipeline_args: list, eval_split: str = None):
    preflight(cfg)
    eff = write_effective_env(cfg)
    job = job_script_path()
    if not job.is_file():
        die(f"packaged job script missing: {job}")
    out_test = str(cfg["_outdir"] / "test")
    args = (_shared_pipeline_args(cfg)
            + ["--out-dir", out_test]
            + pipeline_args)
    log_tmpl = str(cfg["_logdir"] / "skinet_$JOB_ID.log")
    # metrics need ground truth -> only for labelled splits (test/all), never run
    vlist = [f"SKINET_ENV={eff}"]
    if eval_split:
        vlist += [f"SKINET_EVAL=1", f"SKINET_EVAL_SPLIT={eval_split}",
                  f"SKINET_EVAL_OUTDIR={out_test}"]
    qsub_cmd = [
        "qsub", "-q", cfg["SKINET_QUEUE"], "-N", "skinet", "-j", "y",
        "-v", ",".join(vlist),
        "-o", log_tmpl,
        str(job), *args,
    ]
    say(f"submitting: {_c('1', label)}")
    try:
        out = subprocess.check_output(qsub_cmd, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        die("qsub failed:\n" + e.output.decode(errors="replace"))
    print("  " + out.strip())
    m = re.search(r"[Yy]our job(?:-array)?\s+(\d+)", out) or re.search(r"\b(\d{3,})\b", out)
    if m:
        jid = m.group(1)
        print()
        ok(f"job {_c('1', jid)} queued")
        print(f"      watch it:   {_c('36', 'skinet watch ' + jid)}")
        print(f"      queue:      {_c('36', 'skinet status')}")
        print(f"      log file:   {cfg['_logdir']}/skinet_{jid}.log")
        print(f"      results in: {out_test}")


# --------------------------------------------------------------------------- #
#  subcommands
# --------------------------------------------------------------------------- #
def cmd_run(cfg, a, extra):
    img = Path(a.slide)
    if not img.is_file(): die(f"file not found: {img}")
    if not os.access(img, os.R_OK): die(f"file not readable (permissions?): {img}")
    src = img.stem
    # a user-supplied image has no ground truth -> no metrics, only overlay+JSON
    submit(cfg, f"run {src}",
           ["--image", str(img), "--source", src, "--split", "test",
            "--overlay", "--tissue-debug"] + extra)


def cmd_test(cfg, a, extra):
    submit(cfg, f"test {a.source_key}  (+ metrics)",
           ["--source", a.source_key, "--split", "test", "--overlay"] + extra,
           eval_split="test")


def cmd_all(cfg, a, extra):
    submit(cfg, "all test sources  (+ metrics)",
           ["--all", "--split", "test", "--overlay"] + extra,
           eval_split="test")


def cmd_bench(cfg, a, extra):
    # benchmark = timing only; skip metrics so eval time doesn't pollute the number
    submit(cfg, "benchmark (tissue filter OFF, no metrics)",
           ["--all", "--split", "test", "--no-tissue-filter"] + extra)


def cmd_eval(cfg, a, extra):
    """Re-run metrics on an existing output dir without re-segmenting."""
    pred_dir = Path(a.pred_dir) if a.pred_dir else (cfg["_outdir"] / "test")
    if not pred_dir.is_dir():
        die(f"no output dir to evaluate: {pred_dir}\n"
            f"      run `skinet test <KEY>` or `skinet all` first")
    if not any(pred_dir.glob("*.json")):
        die(f"no result JSON in {pred_dir} -- nothing to evaluate")
    preflight(cfg)
    eff = write_effective_env(cfg)
    job = job_script_path()
    log_tmpl = str(cfg["_logdir"] / "skinet_$JOB_ID.log")
    # a no-op pipeline call (image already segmented): we only want the metrics
    # block in the job script, so submit with a sentinel that skips e7_run.
    vlist = [f"SKINET_ENV={eff}", "SKINET_EVAL=1",
             f"SKINET_EVAL_SPLIT={a.split}", f"SKINET_EVAL_OUTDIR={pred_dir}",
             "SKINET_EVAL_ONLY=1"]
    qsub_cmd = ["qsub", "-q", cfg["SKINET_QUEUE"], "-N", "skinet-eval", "-j", "y",
                "-v", ",".join(vlist), "-o", log_tmpl, str(job)]
    say(f"submitting: {_c('1', 'eval ' + str(pred_dir))}")
    try:
        out = subprocess.check_output(qsub_cmd, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        die("qsub failed:\n" + e.output.decode(errors="replace"))
    print("  " + out.strip())
    m = re.search(r"[Yy]our job(?:-array)?\s+(\d+)", out)
    if m:
        ok(f"job {_c('1', m.group(1))} queued -- metrics -> {pred_dir}/metrics")


def cmd_status(cfg, a, extra):
    if not shutil.which("qsub"):
        die("qsub not found -- are you on a CCuB login node?")
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    say("your jobs:")
    try:
        subprocess.run(["qstat", "-u", user or "*"], check=False)
    except FileNotFoundError:
        warn("qstat not found")


def cmd_watch(cfg, a, extra):
    log = cfg["_logdir"] / f"skinet_{a.job_id}.log"
    if not log.is_file():
        warn(f"log not there yet: {log}")
        warn("the job is probably still WAITING. Check: skinet status")
        return
    say(f"tailing {log}   (Ctrl-C stops watching; the job keeps running)")
    print("-" * 66)
    try:
        subprocess.run(["tail", "-f", str(log)], check=False)
    except KeyboardInterrupt:
        pass


def cmd_where(cfg, a, extra):
    print("  config file :", cfg["_config_file"])
    print("  " + _c("1", "shared (read-only, called by path)"))
    print("    scripts   :", cfg["SKINET_SCRIPTS_DIR"])
    print("    conda env :", cfg["SKINET_CONDA_ENV"], "via", cfg["SKINET_CONDA_SH"])
    print("    M2F lib   :", cfg["SKINET_M2F_LIB"])
    print("    SAM lib   :", cfg["SKINET_SAM_LIB"])
    print("    weights   :", cfg["SKINET_WEIGHTS"])
    print("    split CSV :", cfg["SKINET_SPLIT_CSV"])
    print("  " + _c("1", "yours (writable)"))
    print("    logs      :", cfg["_logdir"] / "skinet_<JOB_ID>.log")
    print("    outputs   :", cfg["_outdir"] / "test")
    print("    queue     :", cfg["SKINET_QUEUE"])


def cmd_doctor(cfg, a, extra):
    say("checking your setup...")
    print("  config:", cfg["_config_file"])
    sdir = Path(cfg["SKINET_SCRIPTS_DIR"])
    (ok if (sdir / "e7_run.py").is_file() else lambda *x: die(f"scripts not found: {sdir}"))(f"scripts: {sdir}")
    for name in ("tissue_filter.py", "e7_config.py"):
        (ok if (sdir / name).is_file() else warn)(f"{name} {'present' if (sdir / name).is_file() else 'missing'}")
    for label, key in (("conda.sh", "SKINET_CONDA_SH"),
                       ("weights", "SKINET_WEIGHTS"),
                       ("split CSV", "SKINET_SPLIT_CSV")):
        p = cfg[key]
        if p:
            (ok if os.access(p, os.R_OK) else warn)(
                f"{label} {'readable' if os.access(p, os.R_OK) else 'NOT readable: ' + p + '  (ask owner for chmod g+rX)'}")
    for label, key in (("M2F lib", "SKINET_M2F_LIB"), ("SAM lib", "SKINET_SAM_LIB")):
        p = cfg[key]
        (ok if os.path.isdir(p) else warn)(
            f"{label} {'present' if os.path.isdir(p) else 'NOT found: ' + p}")
    try:
        cfg["_outdir"].mkdir(parents=True, exist_ok=True)
        ok(f"work folder writable: {cfg['_workdir']}")
    except Exception:
        warn(f"cannot write to {cfg['_workdir']} (set SKINET_WORKDIR)")
    (ok if shutil.which("qsub") else warn)(
        f"qsub {'available' if shutil.which('qsub') else 'not found (not on a login node?)'}")
    # Mask2Former base config is read at model build time from e7_config.BASE_CONFIG;
    # surface it here so a missing/unreadable config is caught before submitting.
    try:
        import sys as _sys
        _sys.path.insert(0, str(sdir))
        import importlib
        _ec = importlib.import_module("e7_config")
        bc = getattr(_ec, "BASE_CONFIG", None)
        pre = getattr(_ec, "PRETRAINED", None)
        if bc is not None:
            (ok if os.access(str(bc), os.R_OK) else warn)(
                f"M2F base config {'readable' if os.access(str(bc), os.R_OK) else 'NOT readable: ' + str(bc)}")
        if pre is not None and not os.access(str(pre), os.R_OK):
            warn(f"M2F pretrained NOT readable: {pre} (only needed if a stage loads it)")
    except Exception as e:
        warn(f"could not read e7_config for BASE_CONFIG check ({e})")


# --------------------------------------------------------------------------- #
#  argument parser
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="skinet",
        description="Submit the SKINET kidney-biopsy pipeline to the CCuB cluster.",
        epilog="Extra flags after the command pass straight to the pipeline, "
               "e.g.  skinet all --sam --sam-mode replace")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    sp = sub.add_parser("run", help="segment ONE slide you point at")
    sp.add_argument("slide", help="path to a slide (.jp2/.jpg/...) you can read")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("test", help="segment ONE source from the test split")
    sp.add_argument("source_key", help="e.g. ML0234B10")
    sp.set_defaults(func=cmd_test)

    sp = sub.add_parser("all", help="segment the whole test split (+ metrics)")
    sp.set_defaults(func=cmd_all)

    sp = sub.add_parser("eval", help="(re)compute metrics on an existing output dir")
    sp.add_argument("pred_dir", nargs="?", default=None,
                    help="dir with result JSONs (default: your last test output)")
    sp.add_argument("--split", default="test", choices=["train", "val", "test"])
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("bench", help="whole test split, tissue filter OFF (timing)")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("status", help="your jobs in the queue")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("watch", help="follow a job's log live")
    sp.add_argument("job_id")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("where", help="print resolved paths")
    sp.set_defaults(func=cmd_where)

    sp = sub.add_parser("doctor", help="check your setup before submitting")
    sp.set_defaults(func=cmd_doctor)

    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # split known args from pass-through pipeline flags
    args, extra = parser.parse_known_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    cfg = load_config()
    args.func(cfg, args, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
