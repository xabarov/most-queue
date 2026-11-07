"""Rebuild the tagged-customer diagram for the concise edition.

Scientific construction and layout are inherited from the full edition;
only the Russian label follows the author's choice: «помеченная».
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COLORS = ["#246b8e", "#e4a647", "#ad4a4a"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "figure.constrained_layout.use": True,
    }
)


def save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"revision_{name}.{suffix}", dpi=190)
    plt.close(fig)


def main():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis("off")
    ax.text(4.7, 4.5, r"Состояние в период ожидания: $(R,i,\varphi,K)$", ha="center", fontsize=12)
    ax.text(1.25, 3.35, r"$R\to R-1$: $R\gamma$", ha="center", bbox=dict(boxstyle="round", fc="#e8f1f5"))
    ax.text(
        4.75,
        3.35,
        r"$K\to K+1$: $\lambda$" + "\n" + r"$K\to K-1$: $K\gamma$",
        ha="center",
        bbox=dict(boxstyle="round", fc="#e8f1f5"),
    )
    ax.text(
        8.25, 3.35, "Уход помеченной\n" + r"заявки: $\gamma$", ha="center", bbox=dict(boxstyle="round", fc="#f4dfdf")
    )
    ax.text(4.7, 2.4, "При завершении текущего пакета", ha="center", fontsize=11)
    for x, text in [
        (1.3, r"$R\geq b$" + "\n" + r"$R\to R-b$" + "\nновый пакет размера " + r"$b$"),
        (4.75, r"$R<b,\ R+1+K\geq a$" + "\nначало обслуживания\nпомеченной заявки"),
        (8.2, r"$R<b,\ R+1+K<a$" + "\nпростой и накопление"),
    ]:
        ax.annotate("", xy=(x, 1.65), xytext=(4.7, 2.15), arrowprops=dict(arrowstyle="->"))
        ax.text(
            x,
            0.7,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=.4", fc="#e8f1f5", ec=COLORS[0]),
        )
    save(fig, "tagged")


if __name__ == "__main__":
    main()
