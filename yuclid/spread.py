from scipy.stats import norm
import seaborn as sns
import pandas as pd
import numpy as np
import re

available = {
    "sd", "piX", "rsdX", "mad", "range", "iqr"
}

def mad(y):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    return np.median(np.abs(y - np.median(y)))

def lower(spread_measure):
    n = re.search(r"\d+(\.\d+)?", spread_measure)
    if spread_measure.startswith("sd"):
        coeff = float(n.group())
        return lambda y: y.mean() - coeff * y.std()
    elif spread_measure.startswith("pi"):
        p = float(n.group()) / 100.0
        return lambda y: np.quantile(y, (1 - p)/2)
    elif spread_measure.startswith("rsd"):
        coeff = float(n.group())
        return lambda y: y.mean() - coeff * (1 / norm.ppf(0.75)) * mad(y)
    elif spread_measure == "mad":
        return lambda y: np.median(y) - mad(y)
    elif spread_measure == "range":
        return lambda y: y.min()
    elif spread_measure == "iqr":
        return lambda y: np.quantile(y, 0.25)

def upper(spread_measure):
    n = re.search(r"\d+(\.\d+)?", spread_measure)
    if spread_measure.startswith("sd"):
        coeff = float(n.group())
        return lambda y: y.mean() + coeff * y.std()
    elif spread_measure.startswith("pi"):
        p = float(n.group()) / 100.0
        return lambda y: np.quantile(y, (1 + p)/2)
    elif spread_measure.startswith("rsd"):
        coeff = float(n.group())
        return lambda y: y.mean() + coeff * (1 / norm.ppf(0.75)) * mad(y)
    elif spread_measure == "mad":
        return lambda y: np.median(y) + mad(y)
    elif spread_measure == "range":
        return lambda y: y.max()
    elif spread_measure == "iqr":
        return lambda y: np.quantile(y, 0.75)

def draw(ax, spread_measures, df, x, y, z=None, colors=None, palette=None, style="area"):
    if z is None:
        z = "z" * (max([len(c) for c in columns]) + 1) # unique column name
        df[z] = 0

    alphas = {
        "area": np.linspace(0.15, 0.05, len(spread_measures)),
        "bar": np.linspace(0.30, 0.10, len(spread_measures))
    }[style]

    x_dom = df[x].unique()
    z_dom = df[z].unique()
    groups = df.groupby([z, x])[y]

    if isinstance(palette, str):
        colors = sns.color_palette(palette)
        palette = None

    if colors is None and palette is None:
        raise Exception("Missing color and palette")

    for i, sm in enumerate(spread_measures):
        ys_lower = groups.apply(lower(sm))
        ys_upper = groups.apply(upper(sm))
        for j, z_val in enumerate(z_dom):
            if colors is not None:
                color = colors[j % len(colors)]
            else:
                color = palette[z_val]
            y_lower = ys_lower.xs(z_val)
            y_upper = ys_upper.xs(z_val)
            if style == "area":
                ax.fill_between(x_dom, y_lower, y_upper,
                                interpolate=True, color=color, alpha=alphas[i])
            elif style == "bar":
                ax.vlines(x=x, ymin=y_lower, ymax=y_upper,
                          color=color, alpha=alphas[i], linewidth=4)
