"""
Energy-arc PNG: target curve + actual tracks over set time, colored by genre.
Dark theme on the curator surface; palette values are the CVD-validated
chart variants in config.GENRE_COLORS.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .sequencer import SetResult


def render_png(result: SetResult, df: pd.DataFrame, path: str) -> None:
    ui = config.UI
    total_s = float(df["start_s"].iloc[-1] + df["playtime_s"].iloc[-1])

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=160)
    fig.patch.set_facecolor(ui["bg"])
    ax.set_facecolor(ui["bg"])

    # phase shading + names (recessive)
    for phase in result.template.phases:
        x0, x1 = phase.t0 * total_s / 3600, phase.t1 * total_s / 3600
        ax.axvspan(x0, x1, color=ui["surface_raised"], alpha=0.35, zorder=0)
        ax.text(
            (x0 + x1) / 2, 0.03, phase.name.replace("_", " "),
            ha="center", va="bottom", color=ui["text_muted"], fontsize=8,
        )

    # target curve
    ts = np.linspace(0, 1, 200)
    target = [result.template.target(t)[0] for t in ts]
    ax.plot(
        ts * total_s / 3600, target,
        color=ui["accent"], lw=2, ls=(0, (4, 3)), zorder=2, label="target arc",
    )

    # tracks
    hours = df["start_s"].to_numpy() / 3600
    for genre in config.GENRES:
        sub = df[df["genre"] == genre]
        if sub.empty:
            continue
        ax.scatter(
            sub["start_s"] / 3600, sub["energy"],
            s=48, color=config.GENRE_COLORS[genre], zorder=3, label=genre,
            edgecolors=ui["bg"], linewidths=0.8,  # surface gap between marks
        )
    bridges = df[df["is_bridge"]]
    if not bridges.empty:
        ax.scatter(
            bridges["start_s"] / 3600, bridges["energy"],
            s=110, facecolors="none", edgecolors=ui["text"], linewidths=1.0,
            zorder=4, label="bridge track",
        )

    ax.set_xlim(0, total_s / 3600)
    ax.set_ylim(0, 1)
    ax.set_xlabel("set time (hours)", color=ui["text_secondary"], fontsize=10)
    ax.set_ylabel("energy", color=ui["text_secondary"], fontsize=10)
    title = result.vibe or result.template.name
    ax.set_title(
        f"{title} — {len(df)} tracks over {total_s / 3600:.1f}h",
        color=ui["text"], fontsize=13, pad=12, loc="left",
    )
    ax.tick_params(colors=ui["text_secondary"], labelsize=9)
    ax.grid(color=ui["border"], lw=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(ui["border"])

    # legend lives below the axes so it can never cover data
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=7,
        facecolor=ui["surface"], edgecolor=ui["border"],
        labelcolor=ui["text"], fontsize=9, framealpha=0.9,
    )
    plt.tight_layout()
    plt.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
