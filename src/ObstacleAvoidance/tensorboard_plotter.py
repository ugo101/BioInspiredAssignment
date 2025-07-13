import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def write_to_csv(
    log_dir,
    output_csv,
    metrics=None
):
    """
    Extract specified scalar metrics from a TensorBoard log directory
    and write them to a CSV file in long format.
    """
    if metrics is None:
        metrics = ["reward/episode"]

    if not os.path.isdir(log_dir):
        raise ValueError(f"Log directory does not exist: {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags().get('scalars', [])
    if not available_tags:
        print(f"[WARNING] No scalar data found in: {log_dir}")
        return

    print("[INFO] Available scalar tags in", log_dir)
    for tag in available_tags:
        print(f" - {tag}")

    selected_tags = [tag for tag in metrics if tag in available_tags]
    if not selected_tags:
        print("[WARNING] None of the specified metrics were found in the log.")
        return

    rows = []
    for tag in selected_tags:
        print(f"[INFO] Extracting tag: {tag}")
        scalar_events = ea.Scalars(tag)
        for event in scalar_events:
            rows.append({
                "step": event.step,
                "tag": tag,
                "value": event.value
            })

    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ["step", "tag", "value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] CSV saved to: {output_csv}")


def plot_tensorboard_stats(
    log_dirs,
    output_csv_dir,
    metrics=None
):
    """
    Accepts a list of log directories. For each one:
      - Extracts TensorBoard scalars to CSV.
      - Loads the CSV.
    Plots 'reward/episode' for each log on the same plot,
    using the folder name as the legend label.
    """
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    plt.figure(figsize=(10, 6))

    for log_dir in log_dirs:
        if not os.path.isdir(log_dir):
            print(f"[WARNING] Log directory does not exist: {log_dir}")
            continue

        # Use log folder name for label
        label = os.path.basename(os.path.normpath(log_dir))
        label = label.replace("phase_1_", "")
        output_csv = os.path.join(output_csv_dir, f"{label}.csv")


        # 1. Write CSV
        write_to_csv(
            log_dir=log_dir,
            output_csv=output_csv,
            metrics=metrics
        )

        # 2. Load CSV
        if not os.path.isfile(output_csv):
            print(f"[WARNING] CSV was not created for {log_dir}")
            continue

        print(f"[INFO] Loading CSV: {output_csv}")
        df = pd.read_csv(output_csv)
        if df.empty:
            print(f"[WARNING] CSV is empty for {log_dir}")
            continue

        # 3. Filter only reward/episode
        reward_df = df[df['tag'] == "reward/episode"]
        if reward_df.empty:
            print(f"[WARNING] No 'reward/episode' data found in {log_dir}")
            continue

        # 4. Plot this log's rewards
        plt.plot(reward_df['step'], reward_df['value'], label=label)

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward per Episode Over Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # User-defined list of paths
    # log_dirs = [
    #     "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/runs/phase_1",
    #     "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/runs/phase_1_lr_1e-4",
    #     "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/runs/phase_1_lr_1e-2"
    # ]
    log_dirs = [
        "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/runs/phase_1_small_nets",
        "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/runs/phase_1_large_nets"
    ]
    output_csv_dir = "/home/damenadmin/Projects/BioInspiredAssignment/src/ObstacleAvoidance/csv_log_files"

    metrics = [
        "reward/episode"
    ]

    plot_tensorboard_stats(
        log_dirs=log_dirs,
        output_csv_dir=output_csv_dir,
        metrics=metrics
    )
