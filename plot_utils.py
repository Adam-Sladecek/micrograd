"""Plot helpers for 2D classification datasets and model boundaries."""

from typing import Any, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
except ModuleNotFoundError:
    plt = None
    np = None
    ListedColormap = None


def _require_matplotlib() -> Tuple[Any, Any, Any]:
    if plt is None or np is None or ListedColormap is None:
        raise ModuleNotFoundError(
            "matplotlib is required. Install with: %pip install matplotlib"
        )
    return plt, np, ListedColormap


def _unpack_by_label(data):
    x0, y0, x1, y1 = [], [], [], []
    for (x1v, x2v), y in data:
        if y == 0:
            x0.append(x1v)
            y0.append(x2v)
        else:
            x1.append(x1v)
            y1.append(x2v)
    return x0, y0, x1, y1


def _axis_bounds(train_data, test_data=None):
    all_data = list(train_data) + (list(test_data) if test_data is not None else [])
    xs = [x1 for (x1, _), _ in all_data]
    ys = [x2 for (_, x2), _ in all_data]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add small padding to keep points away from plot edges.
    x_pad = max((x_max - x_min) * 0.08, 0.1)
    y_pad = max((y_max - y_min) * 0.08, 0.1)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def plot_points(train_data, test_data=None):
    """Plot 2D labeled points with split-based markers (train=o, test=x)."""
    plt, _, _ = _require_matplotlib()

    plt.figure(figsize=(6, 6))

    tr_x0, tr_y0, tr_x1, tr_y1 = _unpack_by_label(train_data)
    plt.scatter(tr_x0, tr_y0, c="tomato", marker="o", s=70, label="train: class 0")
    plt.scatter(
        tr_x1, tr_y1, c="royalblue", marker="o", s=70, label="train: class 1"
    )

    if test_data is not None:
        te_x0, te_y0, te_x1, te_y1 = _unpack_by_label(test_data)
        plt.scatter(te_x0, te_y0, c="tomato", marker="x", s=90, label="test: class 0")
        plt.scatter(
            te_x1,
            te_y1,
            c="royalblue",
            marker="x",
            s=90,
            label="test: class 1",
        )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("2D Dataset")
    x_min, x_max, y_min, y_max = _axis_bounds(train_data, test_data)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", "box")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()


def plot_decision_boundary(model, train_data, test_data=None, threshold=0.5, n=300):
    plt, np, ListedColormap = _require_matplotlib()

    x_min, x_max, y_min, y_max = _axis_bounds(train_data, test_data)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n),
        np.linspace(y_min, y_max, n),
    )

    zz = np.zeros_like(xx, dtype=float)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = model([float(xx[i, j]), float(yy[i, j])]).data

    # Hard class regions (0/1) for clean red/blue areas.
    cls = (zz >= threshold).astype(int)

    # Boundary = grid points where neighboring class labels differ.
    edge = np.zeros_like(cls, dtype=bool)
    edge[:, 1:] |= cls[:, 1:] != cls[:, :-1]
    edge[1:, :] |= cls[1:, :] != cls[:-1, :]

    plt.figure(figsize=(6, 6))
    cmap = ListedColormap(["#f6b0aa", "#aec4ff"])
    plt.contourf(xx, yy, cls, levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=0.85)
    plt.contour(xx, yy, zz, levels=[threshold], colors="black", linewidths=1.2)

    # Overlay points (train=o, test=x), colors by class.
    tr_x0, tr_y0, tr_x1, tr_y1 = _unpack_by_label(train_data)
    plt.scatter(tr_x0, tr_y0, c="tomato", marker="o", s=70, label="train: class 0")
    plt.scatter(tr_x1, tr_y1, c="royalblue", marker="o", s=70, label="train: class 1")

    if test_data is not None:
        te_x0, te_y0, te_x1, te_y1 = _unpack_by_label(test_data)
        plt.scatter(te_x0, te_y0, c="tomato", marker="x", s=90, label="test: class 0")
        plt.scatter(te_x1, te_y1, c="royalblue", marker="x", s=90, label="test: class 1")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.grid(alpha=0.2)
    plt.legend(loc="upper right", fontsize=8)
    plt.show()
