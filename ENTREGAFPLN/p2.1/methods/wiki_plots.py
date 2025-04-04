import numpy as np
import json
from visualize import visualize_tsne_embeddings_wikipedia, visualize_all_tsne_embeddings_wikipedia
from dataset_reader import load_target_words


def plot_wikipedia_embeddings(config):
    """
    Genera visualizaciones t-SNE para embeddings de Wikipedia.
    
    Args:
        config (dict): Configuración con parámetros para la visualización.
            Debe contener:
            - embeddings_path: Ruta al archivo npz de embeddings sin ajustar
            - embeddings_fit_path: Ruta al archivo npz de embeddings ajustados
            - target_words_path: Ruta al archivo de palabras objetivo
            - num_other_words: Número de palabras adicionales para visualizar
            - output_prefix: Prefijo para los archivos de salida
    """
    # Cargar los datos de embeddings
    data_no_fit = np.load(config['embeddings_path'])
    data_fit = np.load(config['embeddings_fit_path'])
    
    # Obtener los embeddings y el word_index
    embeddings_no_fit = data_no_fit['embeddings']
    embeddings_fit = data_fit['embeddings']
    
    # Cargar los índices de palabras
    word_index_no_fit = json.loads(data_no_fit['word_index'].item())
    word_index_fit = json.loads(data_fit['word_index'].item())
    
    # Cargar palabras objetivo
    target_words = load_target_words(config['target_words_path'])
    
    # Preparar palabras para visualización
    all_words_in_embeddings = list(word_index_fit.keys())
    other_words = [w for w in all_words_in_embeddings if w not in target_words][:config['num_other_words']]
    words_to_plot = list(target_words) + other_words
    
    # Generar las visualizaciones para embeddings ajustados
    print(f"Generando visualizaciones para embeddings ajustados (Fit)...")
    
    # Visualización solo de palabras objetivo
    visualize_tsne_embeddings_wikipedia(
        words=target_words,
        embeddings=embeddings_fit,
        word_index=word_index_fit,
        filename=f"{config['output_prefix']}_tsne_plot_Fit.png"
    )
    
    # Visualización de todas las palabras
    visualize_all_tsne_embeddings_wikipedia(
        embeddings=embeddings_fit,
        word_index=word_index_fit,
        words_to_plot=words_to_plot,
        words_to_label=target_words,
        filename=f"{config['output_prefix']}_tsne_plot_All_Fit.png"
    )
    
    # Generar las visualizaciones para embeddings sin ajustar
    print(f"Generando visualizaciones para embeddings sin ajustar (NoFit)...")
    
    # Visualización solo de palabras objetivo
    visualize_tsne_embeddings_wikipedia(
        words=target_words,
        embeddings=embeddings_no_fit,
        word_index=word_index_no_fit,
        filename=f"{config['output_prefix']}_tsne_plot_NoFit.png"
    )
    
    # Visualización de todas las palabras
    visualize_all_tsne_embeddings_wikipedia(
        embeddings=embeddings_no_fit,
        word_index=word_index_no_fit,
        words_to_plot=words_to_plot,
        words_to_label=target_words,
        filename=f"{config['output_prefix']}_tsne_plot_All_NoFit.png"
    )
    
    print(f"Visualizaciones completadas y guardadas con prefijo: {config['output_prefix']}")


if __name__ == "__main__":
    # Configuración para la visualización
    config = {
        'embeddings_path': 'wiki_embeddings.npz',
        'embeddings_fit_path': 'wiki_embeddingsFit.npz',
        'target_words_path': 'target_words_text8.txt',
        'num_other_words': 1000,
        'output_prefix': 'wiki'
    }
    
    # Ejecutar la visualización
    plot_wikipedia_embeddings(config)