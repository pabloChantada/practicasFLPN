import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_similarities(target_words, word_index, embedding_weights, top_n=10):
    """
    Calcula las similitudes de coseno entre palabras objetivo y todo el vocabulario.
    
    Para cada palabra objetivo, encuentra las palabras más similares según
    la similitud de coseno de sus embeddings.
    
    Args:
        target_words (list): Lista de palabras objetivo para analizar.
        word_index (dict): Mapeo de palabras a sus índices en la matriz de embeddings.
        embedding_weights (numpy.ndarray): Matriz de pesos de embeddings.
        top_n (int): Número de palabras similares a retornar para cada palabra objetivo.
        
    Returns:
        dict: Diccionario con palabras objetivo como claves y listas de
              (palabra_similar, similitud) como valores.
    """
    results = {}
    
    # Crear mapeo inverso índice -> palabra
    index_to_word = {i: word for word, i in word_index.items()}
    
    # Número total de palabras en el vocabulario
    vocab_size = len(word_index) + 1
    
    # Calcular todas las similitudes de coseno de una vez (más eficiente)
    all_similarities = cosine_similarity(embedding_weights)
    
    for target_word in target_words:
        if target_word not in word_index:
            print(f"Advertencia: '{target_word}' no está en el vocabulario.")
            continue
        
        # Obtener el índice de la palabra objetivo
        target_idx = word_index[target_word]
        
        # Obtener las similitudes para esta palabra
        similarities = all_similarities[target_idx]
        
        # Ordenar por similitud (de mayor a menor)
        indices = np.argsort(similarities)[::-1]
        
        # Seleccionar las top_n palabras más similares (excluyendo la propia palabra objetivo)
        most_similar = []
        for idx in indices:
            # Ignorar el índice 0 (padding) y la propia palabra
            if idx != target_idx and idx != 0 and idx < vocab_size:
                word = index_to_word.get(idx)
                if word:  # Asegurarse de que el índice corresponde a una palabra
                    most_similar.append((word, similarities[idx]))
                    if len(most_similar) == top_n:
                        break
        
        results[target_word] = most_similar
    
    return results


def save_cosine_similarities(results, filename):
    """
    Guarda las similitudes de coseno en un archivo de texto.

    Args:
        results (dict): Diccionario con palabras objetivo y sus palabras más similares.
        filename (str): Ruta del archivo donde se guardarán los resultados.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for target_word, similar_words in results.items():
            file.write(f"Palabras más similares a '{target_word}':\n")
            for word, similarity in similar_words:
                file.write(f"  {word}: {similarity:.4f}\n")
            file.write("\n")