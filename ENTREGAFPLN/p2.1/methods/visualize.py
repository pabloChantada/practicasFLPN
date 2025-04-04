import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from cosine_sim import main as visualize_args
import numpy as np

def visualize_tsne_embeddings_wikipedia(words, embeddings, word_index, filename=None):
    """
    Visualizes t-SNE embeddings of selected words.

    Args:
        words (list): List of words to visualize.
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Filter the embeddings for the selected words
    indices = [word_index[word] for word in words]
    selected_embeddings = embeddings[indices]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5,len(words) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')

    # Save or display the plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def visualize_all_tsne_embeddings_wikipedia(embeddings, word_index, words_to_plot, words_to_label=None, filename=None):
    """
    Visualizes t-SNE embeddings of selected words with optional labeling.

    Args:
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        words_to_plot (list): List of words to plot.
        words_to_label (list, optional): List of words to label. Defaults to None.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Create a reverse mapping from index to word
    index_word = {index: word for word, index in word_index.items()}

    # Ensure words_to_label is a subset of words_to_plot
    if words_to_label is None:
        words_to_label = words_to_plot
    words_to_label = set(words_to_label).intersection(words_to_plot)

    # Filter the embeddings for the words to plot
    indices_to_plot = [word_index[word] for word in words_to_plot if word in word_index]
    selected_embeddings = embeddings[indices_to_plot]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5,len(words_to_plot) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(12, 12))
    for i, index in enumerate(indices_to_plot):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], alpha=0.5)
        if index_word[index] in words_to_label:  # Annotate only selected words
            plt.annotate(index_word[index],
                         xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
            
    if filename:
        plt.savefig(filename)
    else:
        plt.show()



'''
Funciones propias para visualizar embeddings de palabras usando t-SNE.
'''
def visualize_tsne_embeddings(words, embeddings, word_index, filename=None, 
                             perplexity=30, random_state=42):
    """
    Visualiza embeddings de palabras seleccionadas en el contexto de todos los embeddings.
    
    Aplica t-SNE a todos los embeddings primero,
    pero solo muestra y etiqueta las palabras especificadas.

    Args:
        words (list): Lista de palabras a visualizar y etiquetar.
        embeddings (numpy.ndarray): Array que contiene todos los embeddings.
        word_index (dict): Mapeo de palabras a sus índices en el array de embeddings.
        filename (str, optional): Archivo para guardar la visualización. 
        perplexity (int): Parámetro de perplexidad para t-SNE.
        random_state (int): Semilla para reproducibilidad.
    """
    # Aplicar t-SNE a todos los embeddings primero (para consistencia)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    all_reduced = tsne.fit_transform(embeddings)
    
    # Filtrar solo las palabras solicitadas
    indices = [word_index[word] for word in words if word in word_index]
    selected_reduced = all_reduced[indices]
    
    # Crear visualización
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        if word in word_index:
            idx = indices[i]
            plt.scatter(all_reduced[idx, 0], all_reduced[idx, 1])
            plt.annotate(word, 
                         xy=(all_reduced[idx, 0], all_reduced[idx, 1]), 
                         xytext=(5, 2),
                         textcoords='offset points', 
                         ha='right', va='bottom')
    
    # Guardar o mostrar el gráfico
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def visualize_all_tsne_embeddings(embeddings, word_index, words_to_plot=None, target_words=None,
                                words_to_label=None, filename=None,
                                perplexity=30, random_state=42):
    """
    Visualiza todos los embeddings con etiquetado selectivo.
    
    Muestra todo el espacio de embeddings pero solo etiqueta
    palabras específicas para mejor visualización.

    Args:
        embeddings (numpy.ndarray): Array que contiene todos los embeddings.
        word_index (dict): Mapeo de palabras a sus índices en el array de embeddings.
        words_to_plot (list, optional): Lista de palabras a graficar. Si es None, 
                                       se grafican todas las palabras.
        target_words (list, optional): Lista de palabras objetivo a resaltar.
        words_to_label (list, optional): Lista de palabras a etiquetar. Si es None,
                                        se usan target_words.
        filename (str, optional): Archivo para guardar la visualización.
        perplexity (int): Parámetro de perplexidad para t-SNE.
        random_state (int): Semilla para reproducibilidad.
    """
    # Aplicar t-SNE a todos los embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    all_reduced = tsne.fit_transform(embeddings)
    
    # Configurar palabras a mostrar
    if words_to_plot is None:
        words_to_plot = list(word_index.keys())
        
    if words_to_label is None:
        words_to_label = target_words
        
    # Crear visualización
    plt.figure(figsize=(12, 12))
    
    # Primero graficar todas las palabras en gris claro (fondo)
    plt.scatter(all_reduced[:, 0], all_reduced[:, 1], c='lightgray', alpha=0.3, s=10)
    
    # Luego graficar y etiquetar las palabras seleccionadas
    for word in words_to_plot:
        if word in word_index and word in words_to_label:
            idx = word_index[word]
            plt.scatter(all_reduced[idx, 0], all_reduced[idx, 1], c='blue', alpha=0.7, s=50)
            plt.annotate(word,
                         xy=(all_reduced[idx, 0], all_reduced[idx, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

    # Guardar o mostrar el gráfico
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()