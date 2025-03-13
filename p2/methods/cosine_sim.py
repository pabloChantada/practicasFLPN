import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from sklearn.metrics.pairwise import cosine_similarity

# Función para leer el dataset
def read_dataset(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        print(f"Error al leer el archivo {path}")
        return ""

# Función para leer las palabras objetivo desde un archivo
def read_target_words(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except:
        print(f"Error al leer el archivo {file_path}")
        return []

# Función para entrenar un modelo de embeddings simple
def train_embedding_model(tokenized_text, vocab_size, embedding_size=64):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_size, 
                        input_length=1,
                        name='embedding_layer'))
    model.compile('adam', 'mse')
    
    # Entrenar el modelo con una entrada dummy (solo para inicializar los pesos)
    # En un caso real, tendríamos que entrenar con datos reales
    dummy_y = np.zeros((len(tokenized_text), embedding_size))
    model.fit(np.array(tokenized_text).reshape(-1, 1), dummy_y, epochs=1, verbose=0)
    
    return model

# Función para calcular la similitud del coseno
def compute_cosine_similarities(target_words, word_index, embedding_weights, top_n=10):
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

# Función principal
def main():
    # Configuración
    dataset_path = "/home/clown/2-semester/practicasFLPN/p2/materiales/the_fellowship_of_the_ring.txt"
    target_words_path = "/home/clown/2-semester/practicasFLPN/p2/materiales/target_words_the_fellowship_of_the_ring.txt"
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
