from biodatasets import load_dataset
import statistics
import matplotlib.pyplot as plt


def get_data():
    pathogen = load_dataset("pathogen")
    X, y = pathogen.to_npy_arrays(input_names=["sequence"], target_names=["class"])
    embed = pathogen.get_embeddings("sequence", "protbert", "cls")
    X, y = X[0], y[0]
    return X, y, embed


def print_statistics(data, plot=False):
    prot_len = [len(i) for i in data]
    print(f"Proteins: {data.shape[0]}")
    print("Protein length:")
    print(f"   mean: {round(statistics.mean(prot_len), 1)}")
    print(f"   median: {statistics.median(prot_len)}")
    print(f"   standard deviation: {round(statistics.stdev(prot_len), 1)}")

    if plot:
        plt.hist(prot_len, bins=30)
        plt.yscale("log")
        plt.ylabel("Count")
        plt.xlabel("Protein length")
        plt.show()


def protein_length_analysis(plot=False):
    X, y, embed = get_data()
    print("=== All proteins ===")
    print_statistics(X, plot)
    prot_len = [len(i) for i in X]
    n_emb = embed.shape[1]
    print(f"Embedding scale: {n_emb}")
    n_over_emb = sum(i > n_emb for i in prot_len)
    print(
        f"   proteins with a length greater than {n_emb}: {n_over_emb} ({round(n_over_emb*100/embed.shape[0], 2)}%)"
    )

    X_human = X[y == 0]
    print("\n=== Human ===")
    print_statistics(X_human, plot)

    X_patho = X[y == 1]
    print("\n=== Pathogens ===")
    print_statistics(X_patho, plot)


if __name__ == "__main__":
    protein_length_analysis()
