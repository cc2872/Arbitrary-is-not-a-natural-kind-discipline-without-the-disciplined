"""Shared publication style for the Paper-2 figures.
Embedded (Type-42) fonts, 300 dpi, consistent sans typography + palette.
Every figure: import figstyle; figstyle.set_pub_style(); save via figstyle.save(fig, name).
"""
import os
import matplotlib as mpl

OUTDIR = "new figures"

# categorical rule palette (fig 2, fig 5) + heatmap anchors (fig 3)
PALETTE = dict(
    vestige="#9aa0a6", environmental="#c8443a", coordination="#2e6f95",
    collapse="#b3402a", keep="#0f6e56", grid_line=(1, 1, 1, 0.30),
    na="#e7e4da", zero="0.55",
)

def set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,                 # crisp raster preview; PDF stays vector
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "pdf.compression": 6,
        "font.family": "serif",
        # Times New Roman is the journal-standard serif; the rest are metric-compatible
        # fallbacks (their presence in an output = the render machine lacked Times).
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "Liberation Serif", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.default": "regular",      # inline math matches body text
        "axes.unicode_minus": True,
        "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.titlepad": 5.0, "axes.labelpad": 3.0,
        "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": 7.5, "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })

def panel_letter(ax, s, x=-0.12, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="right")

def save(fig, name, title=None):
    os.makedirs(OUTDIR, exist_ok=True)
    meta = {"Title": title or name, "Creator": "matplotlib", "Producer": "Paper-2 figure pipeline"}
    fig.savefig(os.path.join(OUTDIR, name + ".pdf"), metadata=meta)
    fig.savefig(os.path.join(OUTDIR, name + ".png"))
    print("wrote", os.path.join(OUTDIR, name + ".pdf"), "+ .png")
