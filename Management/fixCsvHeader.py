#!/usr/bin/env python3
import os
import csv
import argparse


def process_file(filepath):
    # Read all rows from the CSV file
    with open(filepath, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
    if not rows:
        return

    # Normalize header: lowercase and replace spaces with underscores
    header = rows[0]
    new_header = [col.strip().lower().replace(" ", "_") for col in header]
    rows[0] = new_header

    # Write back all rows with the updated header
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def main(folder):
    # Process each CSV file in the specified folder
    for filename in os.listdir(folder):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(folder, filename)
            process_file(filepath)
            print(f"Processed {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix CSV headers: replace spaces with underscores and lowercase all letters."
    )
    parser.add_argument("folder", help="Path to the folder containing CSV files")
    args = parser.parse_args()
    main(args.folder)
