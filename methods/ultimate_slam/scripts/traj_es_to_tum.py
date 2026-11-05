#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_tum")
    args = parser.parse_args()

    rows = []
    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            stamp_ns = float(row["timestamp"])
            rows.append(
                (
                    stamp_ns / 1e9,
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                )
            )

    output = Path(args.output_tum)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(
                f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f} {row[3]:.9f} "
                f"{row[4]:.9f} {row[5]:.9f} {row[6]:.9f} {row[7]:.9f}\n"
            )


if __name__ == "__main__":
    main()
