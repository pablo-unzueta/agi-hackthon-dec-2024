import torch
import numpy as np
from test import extract_esm_embeddings


def process_aa_seq_file(filepath):
    """Process aa-seq.txt and extract sequences"""
    sequences = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    name = parts[0]
                    seq = parts[1]
                    sequences.append((name, seq))
    return sequences


def save_embeddings(sequences, output_path="embeddings.npy"):
    """Extract and save embeddings"""
    # Get embeddings
    embeddings_dict = extract_esm_embeddings(sequences)

    # Convert embeddings to numpy arrays and flatten
    names = []
    flattened_embeddings = []
    for name, emb in embeddings_dict.items():
        names.append(name)
        flattened_embeddings.append(
            emb.mean(dim=0).numpy()
        )  # Average over sequence length

    X = np.array(flattened_embeddings)

    # Save embeddings and names
    np.save(output_path, X)
    with open("sequence_names.txt", "w") as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"Embeddings saved to {output_path}")
    print(f"Sequence names saved to sequence_names.txt")


if __name__ == "__main__":
    sequences = process_aa_seq_file("aa-seq.txt")
    save_embeddings(sequences)
