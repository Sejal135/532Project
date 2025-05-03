import re
import matplotlib.pyplot as plt
import numpy as np
import sys


def extract_throughputs(log_file_path):
    """
    Extract throughput values from a log file.
    The log contains lines like:
    "Throughput: 123.45 requests/sec"
    """
    throughputs = []
    line_numbers = []  # To track the position in the log (as a proxy for time)
    line_count = 0

    # Compile regex pattern to match throughput lines
    pattern = r"Throughput:\s+(\d+\.\d+)\s+requests/sec"

    try:
        with open(log_file_path, "r") as file:
            for line in file:
                line_count += 1
                match = re.search(pattern, line)
                if match:
                    throughput = float(match.group(1))
                    throughputs.append(throughput)
                    line_numbers.append(line_count)

        if not throughputs:
            print("No throughput data found in the log file.")
            return None, None

        return throughputs, line_numbers
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading log file: {e}")
        return None, None


def plot_throughput(throughputs, line_numbers=None):
    """
    Create a plot of throughput values over time.
    """
    if throughputs is None or len(throughputs) == 0:
        return

    # If line numbers not provided, use sequential indices
    x_values = line_numbers if line_numbers else range(1, len(throughputs) + 1)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        x_values, throughputs, marker="o", linestyle="-", color="blue", markersize=4
    )

    # Calculate moving average for smoother trend line (window size of 5 or less if fewer points)
    window_size = min(5, len(throughputs))
    if window_size > 1:
        moving_avg = np.convolve(
            throughputs, np.ones(window_size) / window_size, mode="valid"
        )
        # Adjust x values for the moving average (centered)
        ma_x = x_values[window_size - 1 :]
        plt.plot(
            ma_x,
            moving_avg,
            linestyle="-",
            color="red",
            linewidth=2,
            label=f"{window_size}-point Moving Average",
        )

    # Add horizontal line at the average throughput
    avg_throughput = sum(throughputs) / len(throughputs)
    plt.axhline(
        y=avg_throughput,
        color="green",
        linestyle="--",
        label=f"Average: {avg_throughput:.2f} req/sec",
    )

    # Add labels and title
    plt.xlabel("Log Entry Number")
    plt.ylabel("Throughput (requests/sec)")
    plt.title("System Throughput Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add some stats as text
    stats_text = f"""
    Max: {max(throughputs):.2f} req/sec
    Min: {min(throughputs):.2f} req/sec
    Avg: {avg_throughput:.2f} req/sec
    """
    plt.annotate(
        stats_text,
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
    )

    # Ensure y-axis starts from 0 for better perspective
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("throughput_plot.png")
    plt.show()


def main():
    # Check if log file path is provided as argument
    if len(sys.argv) > 1:
        log_file_path = sys.argv[1]
    else:
        log_file_path = input("Enter the path to your log file: ")

    throughputs, line_numbers = extract_throughputs(log_file_path)

    if throughputs:
        print(f"Extracted {len(throughputs)} throughput values")
        plot_throughput(throughputs, line_numbers)


if __name__ == "__main__":
    main()
