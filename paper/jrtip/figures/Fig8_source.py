"""Generate the YOLOv8n versus SRPA-YOLO missed-target comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _detection_figure_utils import configure_matplotlib, draw_pair, load_cases, save_figure


SELECTED_IMAGES = ["02100.jpg", "00759.jpg", "00510.jpg"]


def main() -> None:
    configure_matplotlib()
    cases = load_cases(SELECTED_IMAGES)
    figure = plt.figure(figsize=(12.6, 4.65), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, hspace=0.07, wspace=0.07)
    for column, case in enumerate(cases):
        draw_pair(figure, grid[0, column], case, prediction=False,
                  title=f"({chr(97 + column)}) YOLOv8n")
        draw_pair(figure, grid[1, column], case, prediction=True,
                  title=f"({chr(100 + column)}) SRPA-YOLO")
    save_figure(figure, "detection_comparison")


if __name__ == "__main__":
    main()
