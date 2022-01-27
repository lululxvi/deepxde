import csv


def dat_to_csv(dat_file_path, csv_file_path, columns):
    """Converts a dat file to CSV format and saves it.

    Args:
        dat_file_path (string): Path of the dat file.
        csv_file_path (string): Desired path of the CSV file.
        columns (list): Column names to be added in the CSV file.
    """
    with open(dat_file_path, "r") as dat_file, open(
        csv_file_path, "w", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(columns)
        for line in dat_file:
            if "#" in line:
                continue
            row = [field.strip() for field in line.split(" ")]
            csv_writer.writerow(row)
