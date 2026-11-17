"""
Three pie charts of where the cost lives: weights, activations, time.

Drawn for a thesis, which means two things. The figure is built at its final
printed size so nothing is rescaled by \\includegraphics -- rescaling is what
turns a readable 9pt label into an unreadable 4pt one. And the figure carries
almost no text: titles and percentages only. Everything that explains the figure
belongs in the LaTeX caption, which this script prints for you to paste.

The two disagree, and that disagreement is the point: the backbone holds most
of the parameters but a smaller share of the runtime, while the pixel decoder
takes a fifth of the time on a twentieth of the parameters.

Activation memory was originally a third chart, but three pies across a 5.7 inch
text width leaves each about 4.5 cm across and unreadable in print. It carries
one number rather than a distribution -- that activations, not weights, account
for most of the memory -- and that belongs in a sentence. Pass --with-activation
to restore it as a third panel.

Usage:
  python fig_cost_pies.py --profile work/profile/baseline_seed42.json
  python fig_cost_pies.py --profile ... --style times --width 5.5
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = {
    "backbone": "Backbone",
    "pixel_decoder": "Pixel decoder",
    "transformer_decoder": "Transformer decoder",
    "proposal_generator": "Proposal generator",
    "roi_heads": "RoI heads",
    "other": "Other",
}
# held constant across all three charts so a component can be followed by eye
COLOR = {
    "backbone": "#4C72B0",
    "pixel_decoder": "#DD8452",
    "transformer_decoder": "#55A868",
    "proposal_generator": "#DD8452",
    "roi_heads": "#55A868",
    "other": "#BBBBBB",
}


def set_style(style, base):
    """Match the document rather than matplotlib's defaults."""
    if style == "cm":            # stock LaTeX
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["cmr10", "CMU Serif", "DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "cm"
    elif style == "times":       # Times-based templates
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman",
                                      "DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "stix"
    else:
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = base
    plt.rcParams["axes.unicode_minus"] = False


def pie(ax, keys, values, title, base):
    keys = [k for k in keys if values.get(k, 0) > 0]
    vals = [values[k] for k in keys]
    if sum(vals) <= 0:
        ax.axis("off")
        return
    ax.pie(vals, labels=[SHORT.get(k, k) for k in keys],
           colors=[COLOR.get(k, "#999999") for k in keys],
           autopct=lambda p: "%.0f%%" % p if p >= 5 else "",
           startangle=90, counterclock=False,
           labeldistance=1.12, pctdistance=0.62,
           textprops={"fontsize": base - 1},
           wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
    ax.set_title(title, fontsize=base, pad=6)
    ax.axis("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"],
                    help="pdf keeps the text as vectors, which is what LaTeX wants")
    ap.add_argument("--style", default="cm", choices=["cm", "times", "sans"],
                    help="cm for stock LaTeX, times for Times-based templates")
    ap.add_argument("--width", type=float, default=6.0,
                    help="final printed width in inches. Measure \\textwidth "
                         "with \\the\\textwidth and divide by 72.27")
    ap.add_argument("--font-size", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--with-activation", action="store_true",
                    help="add activation memory as a third panel")
    args = ap.parse_args()

    with open(args.profile) as f:
        rep = json.load(f)
    comps = rep["components"]
    order = list(comps.keys())

    weights = dict((k, comps[k].get("weights_mb", 0.0)) for k in order)
    act = dict((k, comps[k].get("act_peak_mb", 0.0)) for k in order)
    ms = dict((k, comps[k].get("ms", 0.0)) for k in order)

    # only the time chart has a measurable remainder
    other = rep["ms_per_tile"] - sum(ms.values())
    order_ms = order + (["other"] if other > 0.5 else [])
    if other > 0.5:
        ms["other"] = other

    set_style(args.style, args.font_size)
    n_panels = 3 if args.with_activation else 2
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(args.width, args.width / (1.95 if
                                      n_panels == 2 else 2.9)))
    pie(axes[0], order, weights, "Parameters (%.0f MB)" % sum(weights.values()),
        args.font_size)
    if args.with_activation:
        pie(axes[1], order, act, "Activation peak", args.font_size)
    pie(axes[-1], order_ms, ms, "Time (%.0f ms/tile)" % rep["ms_per_tile"],
        args.font_size)
    fig.tight_layout(pad=0.3, w_pad=1.2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.profile).stem
    fig_path = out_dir / ("cost_pies_%s.%s" % (stem, args.fmt))
    fig.savefig(str(fig_path), dpi=args.dpi, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)

    csv_path = out_dir / ("cost_pies_%s.csv" % stem)
    tot_w = sum(weights.values())
    with open(str(csv_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "params", "weights_mb", "weights_pct",
                    "act_peak_mb", "ms_per_tile", "time_pct"])
        for k in order_ms:
            c = comps.get(k, {})
            w.writerow([k, c.get("params", ""), round(weights.get(k, 0.0), 1),
                        round(100 * weights.get(k, 0.0) / tot_w, 1) if tot_w else "",
                        round(act.get(k, 0.0), 1), round(ms.get(k, 0.0), 2),
                        round(100 * ms.get(k, 0.0) / rep["ms_per_tile"], 1)])

    prec = "half" if rep.get("fp16") else "full"
    print("wrote %s\nwrote %s\n" % (fig_path, csv_path))
    print("Built at %.1f in wide, so include it without scaling:" % args.width)
    print("-" * 68)
    print(r"\begin{figure}[t]")
    print(r"  \centering")
    print(r"  \includegraphics{figures/%s}" % fig_path.name)
    print(r"  \caption{Distribution of inference cost across the three main")
    print(r"  components of Mask2Former (Swin-B backbone, %d\,px tiles, %s"
          % (rep.get("tile_size", 0), prec))
    print(r"  precision, single tile, batch size~1). The two views do not")
    print(r"  agree: the backbone holds %.0f\%% of the parameters but only"
          % (100 * weights.get("backbone", 0) / tot_w if tot_w else 0))
    print(r"  %.0f\%% of the runtime, while the pixel decoder accounts for"
          % (100 * ms.get("backbone", 0) / rep["ms_per_tile"]))
    print(r"  %.0f\%% of the time on %.0f\%% of the parameters.}"
          % (100 * ms.get("pixel_decoder", 0) / rep["ms_per_tile"],
             100 * weights.get("pixel_decoder", 0) / tot_w if tot_w else 0))
    print(r"  \label{fig:cost-breakdown-%s}" % ("fp16" if rep.get("fp16") else "fp32"))
    print(r"\end{figure}")
    print("-" * 68)


if __name__ == "__main__":
    main()