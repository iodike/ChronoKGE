"""
T-SNE
"""

# import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def t_sne(test_embeddings, test_predictions):
    """"""
    fig, ax = plt.subplots(figsize=(8, 8))

    tsne = TSNE(3, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)
    cmap = cm.get_cmap('tab20')
    num_categories = 10
    for lab in range(num_categories):
        indices = test_predictions == lab
        ax.scatter(tsne_proj[indices, 0],
                   tsne_proj[indices, 1],
                   tsne_proj[indices, 2],
                   c=np.array(cmap(lab)).reshape(1, 4),
                   label=lab,
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

    return
