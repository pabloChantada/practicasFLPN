import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model



# Función para calcular la similitud del coseno
def compute_cosine_similarities(target_words, word_index, embedding_weights, top_n=10):
    """
    Calcula las similitudes de coseno entre las palabras objetivo y todas las demás palabras en el vocabulario.

    Args:
        target_words (list): Lista de palabras objetivo.
        word_index (dict): Diccionario que mapea palabras a índices.
        embedding_weights (numpy.ndarray): Pesos de la capa de embedding (vocab_size, embedding_size).
        top_n (int): Número de palabras más similares a devolver para cada palabra objetivo.

    Returns:
        dict: Un diccionario donde las claves son las palabras objetivo y los valores son listas de tuplas (palabra, similitud).
    """
    results = {}
    
    # Crear mapeo inverso index -> word
    index_to_word = {i: word for word, i in word_index.items()}
    
    # Número total de palabras en el vocabulario
    vocab_size = len(word_index) + 1
    
    # Calcular todas las similitudes de coseno de una vez
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
        results (dict): Diccionario con las palabras objetivo y sus palabras más similares.
        filename (str): Ruta del archivo donde se guardarán los resultados.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for target_word, similar_words in results.items():
            file.write(f"Palabras más similares a '{target_word}':\n")
            for word, similarity in similar_words:
                file.write(f"  {word}: {similarity:.4f}\n")
            file.write("\n")
# Función principal
def main():
    # Configuración
    dataset_path = "materiales/the_fellowship_of_the_ring.txt"
    target_words_path = "../materiales/target_words_the_fellowship_of_the_ring.txt"
    embedding_size = 64
    
    # Leer el dataset
    text = read_dataset(dataset_path)
    if not text:
        return
    
    # Tokenizar el texto
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    
    # Convertir el texto a secuencia de tokens
    tokenized_text = tokenizer.texts_to_sequences([text])[0]
    
    # Entrenar un modelo de embeddings simple
    model = train_embedding_model(tokenized_text, vocab_size, embedding_size)
    
    # Obtener los pesos de la capa de embedding
    embedding_layer = model.get_layer(name="embedding_layer")
    embedding_weights = embedding_layer.get_weights()[0]
    
    # Leer las palabras objetivo
    target_words = read_target_words(target_words_path)
    if not target_words:
        return
    
    # Calcular similitudes
    similarities = compute_cosine_similarities(target_words, word_index, embedding_weights)
    
    # Mostrar resultados
    for target_word, similar_words in similarities.items():
        print(f"\nPalabras más similares a '{target_word}':")
        for i, (word, similarity) in enumerate(similar_words, 1):
            print(f"{i}. {word} (Similitud: {similarity:.4f})")

    return target_words, embedding_weights, word_index

if __name__ == "__main__":
    main()
