"""Generate successful SRPA-YOLO tiny-UAV detections with enlarged views."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _detection_figure_utils import configure_matplotlib, draw_pair, load_cases, save_figure


SELECTED_IMAGES = ["02100.jpg", "00759.jpg", "00470.jpg", "00622.jpg"]
SCENES = ["Sky and vegetation", "Complex vegetation", "Buildings", "Low contrast"]


def main() -> None:
    configure_matplotlib()
    cases = load_cases(SELECTED_IMAGES)
    figure = plt.figure(figsize=(12.0, 5.35), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, hspace=0.08, wspace=0.08)
    for index, case in enumerate(cases):
        title = f"({chr(97 + index)}) {SCENES[index]}"
        draw_pair(figure, grid[index // 2, index % 2], case, prediction=True, title=title,
                  show_ground_truth=False)
    save_figure(figure, "srpa_tiny_uav_examples")


if __name__ == "__main__":
    main()
