import argparse
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_rewards(log_path):
    epochs = []
    rewards = []

    pattern = re.compile(
        r"epoch\s+(\d+)\s+done\s+\|\s+mean reward/FE\s+=\s+([-+]?\d*\.?\d+)",
        re.IGNORECASE,
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)

            if match:
                epochs.append(int(match.group(1)))
                rewards.append(float(match.group(2)))

    return epochs, rewards


def moving_average(values, window):
    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to training log file",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving average window",
    )

    args = parser.parse_args()

    epochs, rewards = parse_rewards(args.log_path)

    print(f"Found {len(rewards)} reward entries")

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs,
        rewards,
        label="Raw reward",
        alpha=0.4,
    )

    if len(rewards) >= args.window:
        smooth_rewards = moving_average(rewards, args.window)

        plt.plot(
            epochs[args.window - 1 :],
            smooth_rewards,
            linewidth=2,
            label=f"Moving average ({args.window})",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Mean reward/FE")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()