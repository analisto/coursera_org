"""
Coursera Catalog — Business Charts Generator
Produces 9 charts in charts/ from data/coursera.csv
"""

import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
DATA_FILE  = ROOT / "data" / "coursera.csv"
CHARTS_DIR = ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BRAND_BLUE    = "#0056D2"   # Coursera blue
BRAND_TEAL    = "#00A0E3"
ACCENT_ORANGE = "#F47B30"
ACCENT_GREEN  = "#2DB37A"
ACCENT_RED    = "#E63946"
ACCENT_PURPLE = "#6A4C93"
GREY          = "#B0B8C1"
BG            = "#FAFBFC"

PALETTE = [BRAND_BLUE, BRAND_TEAL, ACCENT_ORANGE, ACCENT_GREEN,
           ACCENT_RED, ACCENT_PURPLE, "#F4A261", "#264653",
           "#E9C46A", "#2A9D8F", "#457B9D", "#A8DADC"]

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.grid":         True,
    "axes.grid.axis":    "x",
    "grid.color":        "#E0E4EA",
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

SAVE_KWARGS = dict(dpi=150, bbox_inches="tight", facecolor=BG)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, name: str) -> None:
    path = CHARTS_DIR / name
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    print(f"  Saved {path.name}")


def hbar(ax, series: pd.Series, color=BRAND_BLUE, label_fmt="{:,.0f}") -> None:
    """Draw a clean horizontal bar chart on ax from a Series (index=labels, values=counts)."""
    bars = ax.barh(series.index, series.values, color=color, height=0.6, zorder=3)
    ax.set_xlim(0, series.values.max() * 1.18)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar in bars:
        w = bar.get_width()
        ax.text(w + series.values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                label_fmt.format(w), va="center", ha="left", fontsize=9, color="#444")


def fetch_partner_names() -> dict[str, str]:
    """Return {id: name} for all Coursera partners."""
    names: dict[str, str] = {}
    start = 0
    while True:
        try:
            r = requests.get(
                f"https://api.coursera.org/api/partners.v1?limit=200&start={start}&fields=name",
                headers={"user-agent": "Mozilla/5.0"}, timeout=15
            )
            d = r.json()
            for p in d.get("elements", []):
                names[p["id"]] = p["name"]
            nxt = d.get("paging", {}).get("next")
            if not nxt:
                break
            start = int(nxt)
        except Exception:
            break
    return names


def parse_workload_hours(text: str) -> float | None:
    """Best-effort parse of workload text into total weekly hours."""
    if not text or not text.strip():
        return None
    t = text.lower()

    # Patterns: "X-Y hours/week", "X hours a week", "X hrs/week"
    m = re.search(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*hours?.*week", t)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2

    m = re.search(r"(\d+\.?\d*)\s*(?:hours?|hrs?).*(?:per|a|/)\s*week", t)
    if m:
        return float(m.group(1))

    # "N weeks of study, X hours a week"
    m = re.search(r"(\d+\.?\d*)\s*(?:hours?|hrs?).*week", t)
    if m:
        return float(m.group(1))

    # Standalone hours (not per week - treat as total, divide by 4 weeks)
    m = re.search(r"(\d+\.?\d*)\s*(?:hours?|hrs?)", t)
    if m:
        val = float(m.group(1))
        return min(val, 40)  # cap wild values

    # Minutes
    m = re.search(r"(\d+)\s*min", t)
    if m:
        return round(float(m.group(1)) / 60, 2)

    return None

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE, dtype=str).fillna("")

# Parse start dates
df["startDate_ms"] = pd.to_numeric(df["startDate"], errors="coerce")
df["start_dt"] = pd.to_datetime(df["startDate_ms"], unit="ms", errors="coerce")
df["start_year"] = df["start_dt"].dt.year

# Explode pipe-separated multi-value columns
def explode_col(df: pd.DataFrame, col: str) -> pd.Series:
    return (
        df[col]
        .str.split("|")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )

print("Fetching partner names from Coursera API...")
partner_map = fetch_partner_names()
print(f"  Fetched {len(partner_map)} partner names.")

# ---------------------------------------------------------------------------
# Chart 1 — Top 15 Subject Domains
# ---------------------------------------------------------------------------
print("\nGenerating charts...")

domain_counts = (
    explode_col(df, "domainIds")
    .str.replace("-", " ").str.title()
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Chart 1 — Course Volume by Subject Domain", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("Number of courses per top-level domain", pad=8, fontsize=11,
             color="#555", fontweight="normal")
hbar(ax, domain_counts, color=BRAND_BLUE)
ax.set_xlabel("Number of Courses")
fig.tight_layout()
save(fig, "01_top_domains.png")

# ---------------------------------------------------------------------------
# Chart 2 — Top 20 Specialisation Areas (Subdomains)
# ---------------------------------------------------------------------------
subdomain_counts = (
    explode_col(df, "subdomainIds")
    .str.replace("-", " ").str.title()
    .value_counts()
    .head(20)
)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Chart 2 — Top 20 Specialisation Areas", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("Course volume by specialisation sub-category", pad=8, fontsize=11,
             color="#555", fontweight="normal")
hbar(ax, subdomain_counts, color=BRAND_TEAL)
ax.set_xlabel("Number of Courses")
fig.tight_layout()
save(fig, "02_top_subdomains.png")

# ---------------------------------------------------------------------------
# Chart 3 — Language Market Reach
# ---------------------------------------------------------------------------
LANG_LABELS = {
    "en": "English", "es": "Spanish", "fr": "French", "pt-BR": "Portuguese (BR)",
    "ar": "Arabic", "zh-CN": "Chinese (Simplified)", "zh-TW": "Chinese (Traditional)",
    "ru": "Russian", "ko": "Korean", "ja": "Japanese", "de": "German",
    "tr": "Turkish", "id": "Indonesian", "hi": "Hindi", "vi": "Vietnamese",
    "th": "Thai", "uk": "Ukrainian", "it": "Italian", "nl": "Dutch", "pl": "Polish",
}

lang_counts = (
    explode_col(df, "primaryLanguages")
    .map(lambda x: LANG_LABELS.get(x, x))
    .value_counts()
    .head(15)
)

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Chart 3 — Catalog Reach by Language", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("Number of courses available per language", pad=8, fontsize=11,
             color="#555", fontweight="normal")

colors = [BRAND_BLUE if v == "English" else BRAND_TEAL for v in lang_counts.index]
hbar(ax, lang_counts, color=colors)
ax.set_xlabel("Number of Courses")
fig.tight_layout()
save(fig, "03_language_reach.png")

# ---------------------------------------------------------------------------
# Chart 4 — Top 20 Content Partners
# ---------------------------------------------------------------------------
partner_counts_raw = (
    explode_col(df, "partnerIds")
    .value_counts()
    .head(20)
)
partner_counts = partner_counts_raw.copy()
partner_counts.index = [
    partner_map.get(pid, f"Partner {pid}") for pid in partner_counts.index
]

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Chart 4 — Top 20 Content Partners", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("Partners ranked by number of courses contributed", pad=8, fontsize=11,
             color="#555", fontweight="normal")
hbar(ax, partner_counts, color=ACCENT_ORANGE)
ax.set_xlabel("Number of Courses")
fig.tight_layout()
save(fig, "04_top_partners.png")

# ---------------------------------------------------------------------------
# Chart 5 — Catalog Growth Over Time (new courses per year, bar)
# ---------------------------------------------------------------------------
year_counts = (
    df[df["start_year"].between(2015, 2026)]
    ["start_year"]
    .astype(int)
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Chart 5 — New Courses Launched Per Year", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("Volume of course launches per calendar year (2015–2026)", pad=8,
             fontsize=11, color="#555", fontweight="normal")

bars = ax.bar(year_counts.index, year_counts.values, color=BRAND_BLUE, width=0.6, zorder=3)
ax.set_xlabel("Year")
ax.set_ylabel("Courses Launched")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xticks(year_counts.index)
ax.grid(axis="y", zorder=0)
ax.grid(axis="x", visible=False)
ax.spines["bottom"].set_visible(True)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 20, f"{int(h):,}",
            ha="center", va="bottom", fontsize=8.5, color="#444")
fig.tight_layout()
save(fig, "05_catalog_growth_by_year.png")

# ---------------------------------------------------------------------------
# Chart 6 — Cumulative Catalog Size
# ---------------------------------------------------------------------------
cumulative = year_counts.cumsum()

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Chart 6 — Cumulative Catalog Size Over Time", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("Total number of courses available on the platform over time", pad=8,
             fontsize=11, color="#555", fontweight="normal")

ax.plot(cumulative.index, cumulative.values, color=BRAND_BLUE, linewidth=2.5,
        marker="o", markersize=6, zorder=3)
ax.fill_between(cumulative.index, cumulative.values, alpha=0.12, color=BRAND_BLUE)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xlabel("Year")
ax.set_ylabel("Total Courses")
ax.set_xticks(cumulative.index)
ax.grid(axis="y", zorder=0)
ax.grid(axis="x", visible=False)
ax.spines["bottom"].set_visible(True)
for x, y in zip(cumulative.index, cumulative.values):
    ax.text(x, y + 150, f"{int(y):,}", ha="center", va="bottom", fontsize=8.5, color="#444")
fig.tight_layout()
save(fig, "06_cumulative_catalog_growth.png")

# ---------------------------------------------------------------------------
# Chart 7 — Domain Mix by Language (stacked bar)
#   Top 8 domains × Top 6 languages (+ Other)
# ---------------------------------------------------------------------------
TOP_LANGS_CODE = ["en", "es", "fr", "ar", "pt-BR", "zh-CN"]
TOP_LANGS_LABEL = [LANG_LABELS.get(l, l) for l in TOP_LANGS_CODE]

# Expand: one row per (course × domain × language)
df_exp = df.copy()
df_exp["domain"] = df_exp["domainIds"].str.split("|").apply(lambda lst: lst[0] if lst else "")
df_exp["lang"]   = df_exp["primaryLanguages"].str.split("|").apply(lambda lst: lst[0] if lst else "")
df_exp["domain"] = df_exp["domain"].str.replace("-", " ").str.title()
df_exp["lang"]   = df_exp["lang"].map(LANG_LABELS).fillna("Other")

top_domains = df_exp["domain"].replace("", pd.NA).dropna().value_counts().head(8).index.tolist()
df_filtered = df_exp[df_exp["domain"].isin(top_domains)].copy()
df_filtered["lang_group"] = df_filtered["lang"].where(
    df_filtered["lang"].isin(TOP_LANGS_LABEL), other="Other"
)

pivot = (
    df_filtered
    .groupby(["domain", "lang_group"])
    .size()
    .unstack(fill_value=0)
)
# Order columns
col_order = [LANG_LABELS[l] for l in TOP_LANGS_CODE if LANG_LABELS[l] in pivot.columns] + \
            (["Other"] if "Other" in pivot.columns else [])
pivot = pivot[col_order]
pivot = pivot.loc[top_domains[::-1]]   # reverse for top-to-bottom on hbar

stacked_colors = PALETTE[:len(col_order)]

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Chart 7 — Language Mix Within Each Subject Domain", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("How multilingual is each subject area?", pad=8,
             fontsize=11, color="#555", fontweight="normal")

left = np.zeros(len(pivot))
for i, col in enumerate(col_order):
    vals = pivot[col].values.astype(float)
    ax.barh(pivot.index, vals, left=left, color=stacked_colors[i], label=col, height=0.6)
    left += vals

ax.set_xlabel("Number of Courses")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)
ax.grid(axis="x")
ax.spines["left"].set_visible(False)
fig.tight_layout()
save(fig, "07_domain_language_mix.png")

# ---------------------------------------------------------------------------
# Chart 8 — Partner Concentration (top 10 vs rest)
# ---------------------------------------------------------------------------
all_partner_counts = explode_col(df, "partnerIds").value_counts()
top10_ids  = all_partner_counts.head(10).index.tolist()
top10_vals = all_partner_counts.head(10).values
rest_val   = all_partner_counts.iloc[10:].sum()

top10_names = [partner_map.get(pid, f"Partner {pid}") for pid in top10_ids]

labels = top10_names + ["All Other Partners"]
values = list(top10_vals) + [rest_val]
colors = [BRAND_BLUE] * 10 + [GREY]

series = pd.Series(values, index=labels)

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Chart 8 — Partner Concentration", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("Share of catalog contributed by top 10 partners vs. all others",
             pad=8, fontsize=11, color="#555", fontweight="normal")
bars = ax.barh(labels[::-1], [v for v in values[::-1]], color=colors[::-1], height=0.6, zorder=3)
ax.set_xlim(0, max(values) * 1.22)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
total = sum(values)
for bar in bars:
    w = bar.get_width()
    pct = w / total * 100
    ax.text(w + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}  ({pct:.1f}%)", va="center", ha="left", fontsize=9, color="#444")
fig.tight_layout()
save(fig, "08_partner_concentration.png")

# ---------------------------------------------------------------------------
# Chart 9 — Domain Growth: Top 5 Domains Over Time (line)
# ---------------------------------------------------------------------------
top5_domains = (
    df_exp["domain"].replace("", pd.NA).dropna()
    .value_counts().head(5).index.tolist()
)

df_timed = df_exp[
    df_exp["domain"].isin(top5_domains) &
    df_exp["start_year"].between(2016, 2025)
].copy()

pivot_growth = (
    df_timed
    .groupby(["start_year", "domain"])
    .size()
    .unstack(fill_value=0)
)[top5_domains]

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Chart 9 — Domain Growth Trends (2016–2025)", y=1.01,
             fontsize=15, fontweight="bold")
ax.set_title("New course launches per year across the top 5 subject domains",
             pad=8, fontsize=11, color="#555", fontweight="normal")

for i, domain in enumerate(top5_domains):
    ax.plot(pivot_growth.index, pivot_growth[domain],
            marker="o", linewidth=2.2, markersize=5,
            color=PALETTE[i], label=domain, zorder=3)

ax.set_xlabel("Year")
ax.set_ylabel("New Courses Launched")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xticks(pivot_growth.index)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(axis="y", zorder=0)
ax.grid(axis="x", visible=False)
ax.spines["bottom"].set_visible(True)
fig.tight_layout()
save(fig, "09_domain_growth_trends.png")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\nAll charts saved to {CHARTS_DIR}")
