import matplotlib.pyplot as plt 

def plot_tsne(real_embedding, simulated_embedding, simulated_name='fake'):
    plt.figure(figsize=(6,6))

    plt.scatter(
                real_embedding[:, 0],
                real_embedding[:, 1],
                c="blue",
                label="real (test)",
                alpha=0.5,
            )
    plt.scatter(
                simulated_embedding[:, 0],
                simulated_embedding[:, 1],
                c="red",
                label=simulated_name,
                alpha=0.5,
            )

    plt.grid(True)
    plt.legend(
                loc="lower left", numpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0)
            )
    
    
def plot_tsne_by_cluster(embedding, data, label="real"):
    plt.figure(figsize=(6,6))
    plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=list(map(int,list(data.obs.cluster.values))),
                label=label,
                alpha=0.5,
            )
    plt.grid(True)
    plt.legend(
                loc="lower left", numpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0)
            )