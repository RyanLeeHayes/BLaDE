#! /usr/bin/env python

# average_nrg_column14.py

def average_energy_from_nrg(file_path):
    energies = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue  # skip empty lines
            columns = line.strip().split()
            if len(columns) < 14:
                continue  # skip lines with too few columns
            try:
                energy = float(columns[13])  # 14th column is index 13
                energies.append(energy)
            except ValueError:
                continue  # skip lines with non-numeric values

    if len(energies) == 0:
        print("No valid energy values found.")
        return None

    avg_energy = sum(energies) / len(energies)
    print(f"Number of samples: {len(energies)}")
    print(f"Average energy (14th column): {avg_energy:.6f}")
    return avg_energy

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python average_nrg_column14.py path/to/file.nrg")
    else:
        average_energy_from_nrg(sys.argv[1])
