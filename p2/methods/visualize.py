import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from cosine_sim import main as visualize_args
import numpy as np


def visualize_tsne_embeddings(words, embeddings, word_index, filename=None, 
                            perplexity=30, random_state=42):
    """
    Visualiza solo las palabras especificadas, pero usando como contexto TODOS los embeddings.
    """
    # Aplicar t-SNE a todos los embeddings primero (para consistencia)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    all_reduced = tsne.fit_transform(embeddings)
    
    # Filtrar solo las palabras solicitadas
    indices = [word_index[word] for word in words if word in word_index]
    selected_reduced = all_reduced[indices]
    
    # Plot
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        if word in word_index:
            plt.scatter(selected_reduced[i, 0], selected_reduced[i, 1])
            plt.annotate(word, 
                         xy=(selected_reduced[i, 0], selected_reduced[i, 1]), 
                         xytext=(5, 2),
                         textcoords='offset points', 
                         ha='right', va='bottom')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_all_tsne_embeddings(embeddings, word_index, words_to_plot=None, 
                                words_to_label=None, filename=None,
                                perplexity=30, random_state=42):
    """
    Visualiza todas las palabras pero solo etiqueta las especificadas.
    Usa los mismos parámetros t-SNE que la primera función para consistencia.
    """
    # Aplicar t-SNE (con los mismos parámetros que la otra función)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    all_reduced = tsne.fit_transform(embeddings)
    
    # Configurar palabras a mostrar
    if words_to_plot is None:
        words_to_plot = list(word_index.keys())
    if words_to_label is None:
        words_to_label = words_to_plot
        
    indices_to_plot = [word_index[word] for word in words_to_plot if word in word_index]
    selected_reduced = all_reduced[indices_to_plot]
    
    # Plot
    plt.figure(figsize=(12, 12))
    
    # Primero graficar todas las palabras en gris
    plt.scatter(all_reduced[:, 0], all_reduced[:, 1], c='lightgray', alpha=0.3, s=10)
    
    # Luego graficar las palabras seleccionadas
    for i, word in enumerate(words_to_plot):
        if word in word_index and word in words_to_label:
            idx = word_index[word]
            plt.scatter(all_reduced[idx, 0], all_reduced[idx, 1], c='blue', alpha=0.7, s=50)
            plt.annotate(word,
                         xy=(all_reduced[idx, 0], all_reduced[idx, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()