import math
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc


def wenxiang(
    sequence: str,
    *,
    color: Literal["hbpa", "polarity", "charge", "hydropathy"] = "hbpa",
    add_legend: bool = True,
    add_amino_acid_label: bool = True,
    step_size: int = 18,
    rotation_per_amino_acid: int = 100,
    implosion_factor: float = 0.0,
    n_amino_acids_resize: int = None,
    ax: plt.Axes = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a helix wheel using the Wenxiang format.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    color : str, optional
        Color scheme to use.
    add_legend : bool, optional
        Whether to add a legend.
    add_amino_acid_label : bool, optional
        Whether to add amino acid labels to each circle.
    step_size : int, optional
        Step size for drawing dotted lines between amino acids. If None, no lines are drawn.
    rotation_per_amino_acid : int, optional
        Angle between consecutive amino acids in degrees.
    implosion_factor : float, optional
        Implosion factor for the spiral shape. Increase to make the spiral more compact.
        Must be between 0 and 1/n_arcs.
    n_amino_acids_resize : int, optional
        Resize the spiral to fit a fixed number of amino acids visually. If None, no resizing is done.
        This parameter only affects the visualization. The spiral will still contain all amino acids.
        Useful for visualizing multiple sequences with different lengths.
    ax : plt.Axes, optional
        Matplotlib axes.

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure.
    ax : plt.Axes
        Matplotlib axes.

    References
    ----------
    Jungk et. al: "3.0: Evolutionary Visualization of α, π, and 3/10 Helices"

    """

    def rotation_to_arc_idx(angle: int) -> int:
        return angle // 180

    def amino_acid_to_rotation(nth_aa: int) -> int:
        return rotation_per_amino_acid * (nth_aa - 1)

    n_aa = len(sequence)

    if n_aa < 2:
        raise ValueError("Sequence must have at least 2 amino acids.")

    # Not a necessary requirement, but the wheel becomes very large otherwise.
    if n_aa > 50:
        raise ValueError("Sequence must have at most 50 amino acids.")

    if rotation_per_amino_acid < 1:
        raise ValueError("Rotation must be at least 1 degree.")

    if (step_size is not None) and (step_size < 1):
        raise ValueError("Step size must be at least 1.")

    if (n_amino_acids_resize is not None) and (n_amino_acids_resize < 2):
        raise ValueError("The number of amino acids used as reference resize size must be at least 2.")

    total_rotation = amino_acid_to_rotation(n_aa)
    n_arcs = rotation_to_arc_idx(total_rotation) + 1

    if (implosion_factor < 0) or (implosion_factor > 1 / n_arcs):
        upper = _floor_with_decimals(1 / n_arcs, decimals=3)
        raise ValueError(f"Implosion must be between 0 and {upper}.")

    n_circles = math.ceil(n_arcs / 2)

    # Resize the spiral to fit a fixed number of AA.
    if n_amino_acids_resize is not None:
        total_rotation_visual = amino_acid_to_rotation(n_amino_acids_resize)
        n_arcs_visual = rotation_to_arc_idx(total_rotation_visual) + 1
        n_circles_visual = math.ceil(n_arcs_visual / 2)
    else:
        n_circles_visual = n_circles

    # Add an extra outer circle arc (+1), so outer AA are always on the canvas.
    circle_gap = 0.5 / (n_circles_visual + 1)
    start_gap = circle_gap * 1.8

    # Adjust the center of the spiral for aligning arcs with implosion.
    center_x = 0.5
    center_y = 0.5

    implosion = circle_gap * implosion_factor
    y_offset = np.arange(n_circles * 2) * implosion / 2
    y_offset[1::2] *= -1
    center_y2 = np.array([center_y - circle_gap / 2, center_y] * n_circles) + y_offset

    df_spiral = pd.DataFrame(
        {
            "start_angle": [90, 270] * n_circles,
            "end_angle": [270, 90] * n_circles,
            "center_x": [center_x] * n_circles * 2,
            "center_y": center_y2.tolist(),
            "radius": [start_gap + i * (circle_gap - i * implosion) for i in range(n_circles * 2)],
        }
    )

    # Adjust the last arc to match the last AA.
    last_angle = (90 - amino_acid_to_rotation(n_aa)) % 360
    df_spiral.iloc[n_arcs - 1, df_spiral.columns.get_loc("end_angle")] = last_angle

    points = []
    for i in range(n_aa):
        rotation = amino_acid_to_rotation(i + 1)

        arc_idx = rotation_to_arc_idx(rotation)
        arc = df_spiral.iloc[arc_idx]

        angle = 90 + rotation
        radius = arc["radius"] / 2
        x = arc["center_x"] - radius * np.cos(np.radians(angle))
        y = arc["center_y"] + radius * np.sin(np.radians(angle))

        points.append((x, y))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    spiral_dist = float(np.linalg.norm(np.array(points[0]) - np.array(points[1])))
    point_max_radius = spiral_dist / 2
    point_radius = point_max_radius * 0.8

    text_size = point_radius * 380

    arc_thickness = point_radius * 20
    step_thickness = point_radius * 60
    point_thickness = 1.3

    # draw arcs
    for i in range(n_arcs):
        v = df_spiral.iloc[i]

        start_angle = v["start_angle"]
        end_angle = v["end_angle"]

        # skip if the arcs angle is 0 degrees.
        if start_angle == end_angle:
            continue

        ax.add_patch(
            Arc(
                (v["center_x"], v["center_y"]),
                v["radius"],
                v["radius"],
                theta1=v["end_angle"],
                theta2=v["start_angle"],
                color="black",
                linewidth=arc_thickness,
                alpha=1,
                zorder=1,
            )
        )

    if step_size is not None:
        for i in range(0, n_aa - step_size):
            x1, y1 = points[i]
            x2, y2 = points[i + step_size]

            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linestyle="dotted",
                linewidth=step_thickness,
                alpha=1,
                zorder=2,
            )

    # draw points
    if color == "hbpa":
        types = _aa_to_hpba()
        types_palette = _hpba_palette()
    elif color == "polarity":
        types = _aa_to_polarity()
        types_palette = _net_polarity_palette()
    elif color == "charge":
        types = _aa_to_charge()
        types_palette = _charge_palette()
    elif color == "hydropathy":
        types = _aa_to_hydropathy()
        types_palette = _hydropathy_palette()
    else:
        raise ValueError("Invalid colors. Must be 'hbpa', 'polarity', 'charge' or 'hydropathy'.")

    labels = list({types[aa] for aa in sequence})
    label_order = list(types_palette.keys())
    labels.sort(key=lambda x: label_order.index(x))
    colors = [types_palette[types[aa]] for aa in sequence]

    for i, (x, y) in enumerate(points):
        circle = plt.Circle(
            (x, y),
            point_radius,
            facecolor=colors[i],
            edgecolor="black",
            zorder=3,
            linewidth=point_thickness,
        )
        ax.add_artist(circle)

        if add_amino_acid_label:
            ax.text(
                x,
                y,
                sequence[i],
                ha="center",
                va="center",
                fontsize=text_size,
                zorder=4,
            )

    if add_legend:
        patches = [mpatches.Patch(color=types_palette[label], label=label) for label in labels if label != "unknown"]
        ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0.5))

    if n_amino_acids_resize is not None:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_aspect("equal")
    ax.set_axis_off()

    return fig, ax


def _floor_with_decimals(x: float, decimals: int) -> float:
    return math.floor(x * 10**decimals) / 10**decimals


def _hpba_palette() -> dict[str, str]:
    return {
        "hydrophobic": "lightcoral",
        "non-polar": "gold",
        "polar": "deepskyblue",
        "basic": "lime",
        "acidic": "mediumorchid",
        "unknown": "white",
    }


def _aa_to_hpba() -> dict[str, str]:
    return {
        "A": "hydrophobic",
        "C": "non-polar",
        "D": "acidic",
        "E": "acidic",
        "F": "hydrophobic",
        "G": "hydrophobic",
        "H": "basic",
        "I": "hydrophobic",
        "K": "basic",
        "L": "hydrophobic",
        "M": "hydrophobic",
        "N": "polar",
        "P": "hydrophobic",
        "Q": "polar",
        "R": "basic",
        "S": "polar",
        "T": "polar",
        "V": "hydrophobic",
        "W": "hydrophobic",
        "Y": "non-polar",
        "X": "unknown",
    }


def _net_polarity_palette() -> dict[str, str]:
    return {
        "acidic": "mediumorchid",
        "basic": "lime",
        "non-polar": "gold",
        "polar": "deepskyblue",
        "unknown": "white",
    }


def _aa_to_polarity():
    return {
        "A": "non-polar",
        "C": "non-polar",
        "D": "acidic",
        "E": "acidic",
        "F": "non-polar",
        "G": "non-polar",
        "H": "basic",
        "I": "non-polar",
        "K": "basic",
        "L": "non-polar",
        "M": "non-polar",
        "N": "polar",
        "P": "non-polar",
        "Q": "polar",
        "R": "basic",
        "S": "polar",
        "T": "polar",
        "V": "non-polar",
        "W": "non-polar",
        "Y": "non-polar",
        "X": "unknown",
    }


def _aa_to_charge():
    return {
        "A": "neutral",
        "C": "neutral",
        "D": "negative",
        "E": "negative",
        "F": "neutral",
        "G": "neutral",
        "H": "positive",
        "I": "neutral",
        "K": "positive",
        "L": "neutral",
        "M": "neutral",
        "N": "neutral",
        "P": "neutral",
        "Q": "neutral",
        "R": "positive",
        "S": "neutral",
        "T": "neutral",
        "V": "neutral",
        "W": "neutral",
        "Y": "neutral",
        "X": "unknown",
    }


def _charge_palette() -> dict[str, str]:
    return {
        "positive": "red",
        "neutral": "gray",
        "negative": "steelblue",
        "unknown": "white",
    }


def _aa_to_hydropathy():
    return {
        "A": "hydrophobic",
        "C": "moderate",
        "D": "hydrophilic",
        "E": "hydrophilic",
        "F": "hydrophobic",
        "G": "hydrophilic",
        "H": "moderate",
        "I": "hydrophobic",
        "K": "hydrophilic",
        "L": "hydrophobic",
        "M": "moderate",
        "N": "hydrophilic",
        "P": "hydrophobic",
        "Q": "hydrophilic",
        "R": "hydrophilic",
        "S": "hydrophilic",
        "T": "hydrophilic",
        "V": "hydrophobic",
        "W": "hydrophobic",
        "Y": "hydrophobic",
        "X": "unknown",
    }


def _hydropathy_palette() -> dict[str, str]:
    return {
        "hydrophobic": "lightcoral",
        "hydrophilic": "royalblue",
        "moderate": "grey",
        "unknown": "white",
    }
