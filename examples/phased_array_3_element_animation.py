"""
Three-element phased-array beam steering animation.

Run from the project root:

    python examples/phased_array_3_element_animation.py

Or from this examples directory:

    python phased_array_3_element_animation.py

This is a compact teaching demo. It shows that a phased array does not try
random phase settings: for a desired steering angle theta_0, it computes the
adjacent-element phase shift and applies [0, Phi, 2Phi] to make signals from
that direction add coherently.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


NUM_ELEMENTS = 3
WAVELENGTH_M = 1.0
ELEMENT_SPACING_M = WAVELENGTH_M / 2.0
WAVENUMBER_RAD_PER_M = 2.0 * np.pi / WAVELENGTH_M
EPSILON = 1e-12

OBSERVATION_ANGLES_DEG = np.linspace(-180.0, 180.0, 1441)
OBSERVATION_ANGLES_RAD = np.deg2rad(OBSERVATION_ANGLES_DEG)
STEERING_ANGLES_DEG = np.linspace(-180.0, 180.0, 361)


def wrap_degrees(angle_deg: np.ndarray | float) -> np.ndarray | float:
    """Wrap phase angles to [-180, 180) degrees for readable labels."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def phase_shift_for_steering_angle(steering_angle_deg: float) -> float:
    """Adjacent-element phase shift Phi = k d sin(theta_0)."""
    steering_angle_rad = np.deg2rad(steering_angle_deg)
    return WAVENUMBER_RAD_PER_M * ELEMENT_SPACING_M * np.sin(steering_angle_rad)


def array_factor(
    observation_angles_rad: np.ndarray,
    adjacent_phase_shift_rad: float,
    num_elements: int = NUM_ELEMENTS,
) -> np.ndarray:
    """Compute AF(theta) = sum_n exp(j n (k d sin(theta) - Phi))."""
    element_indices = np.arange(num_elements)
    spatial_phase_rad = (
        WAVENUMBER_RAD_PER_M
        * ELEMENT_SPACING_M
        * np.sin(observation_angles_rad)
    )
    phase_error = spatial_phase_rad[:, None] - adjacent_phase_shift_rad
    return np.sum(np.exp(1j * element_indices[None, :] * phase_error), axis=1)


def normalized_gain_db(array_factor_values: np.ndarray) -> np.ndarray:
    """Return 20 log10(|AF| / N), clipped by EPSILON to avoid log(0)."""
    normalized_magnitude = np.abs(array_factor_values) / NUM_ELEMENTS
    return 20.0 * np.log10(np.maximum(normalized_magnitude, EPSILON))


def phase_settings_deg(adjacent_phase_shift_rad: float) -> np.ndarray:
    """Return [0, Phi, 2Phi] in wrapped degrees."""
    phases_rad = np.arange(NUM_ELEMENTS) * adjacent_phase_shift_rad
    return wrap_degrees(np.rad2deg(phases_rad))


def main() -> None:
    element_positions = (
        np.arange(NUM_ELEMENTS) - (NUM_ELEMENTS - 1) / 2.0
    ) * ELEMENT_SPACING_M

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.8, 2.2])
    ax_elements = fig.add_subplot(grid[0])
    ax_text = fig.add_subplot(grid[1])
    ax_gain = fig.add_subplot(grid[2])

    fig.suptitle("3-Element Uniform Linear Array Beam Steering", fontsize=14)

    # Subplot 1: antenna positions and current phase settings.
    ax_elements.scatter(element_positions, np.zeros(NUM_ELEMENTS), s=180, color="tab:blue")
    ax_elements.axhline(0.0, color="0.75", linewidth=1.0)
    ax_elements.set_xlim(element_positions[0] - ELEMENT_SPACING_M, element_positions[-1] + ELEMENT_SPACING_M)
    ax_elements.set_ylim(-0.7, 0.9)
    ax_elements.set_yticks([])
    ax_elements.set_xlabel("Element position on x-axis (meters)")
    ax_elements.set_title("Antenna Elements and Phase Settings")
    ax_elements.grid(True, axis="x", alpha=0.25)

    phase_labels = []
    for index, x_position in enumerate(element_positions):
        ax_elements.text(
            x_position,
            -0.22,
            f"Element {index}",
            ha="center",
            va="top",
            fontsize=9,
        )
        label = ax_elements.text(
            x_position,
            0.32,
            "",
            ha="center",
            va="bottom",
            fontsize=11,
            color="tab:blue",
            fontweight="bold",
        )
        phase_labels.append(label)

    # Subplot 2: numeric state.
    ax_text.axis("off")
    info_text = ax_text.text(
        0.02,
        0.55,
        "",
        transform=ax_text.transAxes,
        ha="left",
        va="center",
        fontsize=12,
        family="monospace",
    )

    # Subplot 3: gain-to-angle / array factor.
    initial_phase_shift = phase_shift_for_steering_angle(STEERING_ANGLES_DEG[0])
    initial_gain_db = normalized_gain_db(
        array_factor(OBSERVATION_ANGLES_RAD, initial_phase_shift)
    )
    (gain_line,) = ax_gain.plot(
        OBSERVATION_ANGLES_DEG,
        initial_gain_db,
        color="tab:blue",
        linewidth=2.0,
    )
    steering_line = ax_gain.axvline(
        STEERING_ANGLES_DEG[0],
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Current steering angle",
    )
    (steering_marker,) = ax_gain.plot(
        [STEERING_ANGLES_DEG[0]],
        [0.0],
        marker="o",
        markersize=7,
        color="tab:red",
    )

    ax_gain.set_xlim(-180.0, 180.0)
    ax_gain.set_ylim(-40.0, 1.0)
    ax_gain.set_xlabel("Observation angle theta (degrees)")
    ax_gain.set_ylabel("Normalized gain (dB)")
    ax_gain.set_title("Array Gain vs Angle")
    ax_gain.grid(True, alpha=0.3)
    ax_gain.legend(loc="lower right")

    def update(frame_index: int):
        steering_angle_deg = STEERING_ANGLES_DEG[frame_index]
        adjacent_phase_shift_rad = phase_shift_for_steering_angle(steering_angle_deg)
        adjacent_phase_shift_deg = np.rad2deg(adjacent_phase_shift_rad)
        phase_degrees = phase_settings_deg(adjacent_phase_shift_rad)

        af = array_factor(OBSERVATION_ANGLES_RAD, adjacent_phase_shift_rad)
        gain_db = normalized_gain_db(af)

        gain_line.set_ydata(gain_db)
        steering_line.set_xdata([steering_angle_deg, steering_angle_deg])

        # At theta = theta_0, kd sin(theta) - Phi = 0, so normalized gain is 0 dB.
        steering_marker.set_data([steering_angle_deg], [0.0])

        for label, phase_deg in zip(phase_labels, phase_degrees):
            label.set_text(f"{phase_deg:+.0f} deg")

        info_text.set_text(
            "\n".join(
                [
                    f"Steering angle theta_0: {steering_angle_deg:+7.1f} deg",
                    (
                        "Adjacent phase shift Phi: "
                        f"{adjacent_phase_shift_rad:+7.3f} rad "
                        f"= {adjacent_phase_shift_deg:+7.1f} deg"
                    ),
                    (
                        "Phase settings [0, Phi, 2Phi]: "
                        f"[{phase_degrees[0]:+.1f}, "
                        f"{phase_degrees[1]:+.1f}, "
                        f"{phase_degrees[2]:+.1f}] deg"
                    ),
                    "Formula: AF(theta) = sum_n exp(j n (k d sin(theta) - Phi))",
                ]
            )
        )

        return [gain_line, steering_line, steering_marker, info_text, *phase_labels]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(STEERING_ANGLES_DEG),
        interval=35,
        blit=False,
        repeat=True,
    )

    # Keep a live reference so Matplotlib does not garbage-collect the animation.
    fig._phased_array_animation = animation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
