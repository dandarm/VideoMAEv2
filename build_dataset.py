import os
import csv


def create_csv(output_dir):
    # File CSV di output
    train_csv = os.path.join(output_dir, "train.csv")
    test_csv = os.path.join(output_dir, "test.csv")
    val_csv = os.path.join(output_dir, "val.csv")

    subfolders = sorted([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])

    total = len(subfolders)
    train_split = int(total * 0.7)
    test_split = int(total * 0.99)

    train_dirs = subfolders[:train_split]
    test_dirs = subfolders[train_split:test_split]
    val_dirs = subfolders[test_split:]

    # Scrive nei file CSV con il formato richiesto
    def write_to_csv(dirs, csv_file):
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["path", "start", "end"])  # Intestazione
            for dir_path in dirs:
                writer.writerow([dir_path, 1, 16])  # Riga nel formato richiesto

    write_to_csv(train_dirs, train_csv)
    write_to_csv(test_dirs, test_csv)
    write_to_csv(val_dirs, val_csv)

    print(f"File CSV generati:\nTrain: {train_csv}\nTest: {test_csv}\nValidation: {val_csv}")