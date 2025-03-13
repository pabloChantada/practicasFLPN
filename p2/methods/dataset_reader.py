import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# Función para leer el archivo y obtener el texto
def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Función para crear secuencias de entrenamiento usando ventana deslizante
def create_training_sequences(tokenized_text, window_size):
    input_sequences = []
    targets = []
    
    for i in range(window_size, len(tokenized_text)):
        # Obtiene la secuencia de la ventana anterior a la posición actual
        window_sequence = tokenized_text[i-window_size:i]
        # La palabra objetivo es la palabra actual
        target_word = tokenized_text[i]
        
        input_sequences.append(window_sequence)
        targets.append(target_word)
    
    return np.array(input_sequences), np.array(targets)

if __name__ == "__main__":
    # Configuración
    dataset_path = "/home/clown/2-semester/practicasFLPN/p2/materiales/the_fellowship_of_the_ring.txt"
    window_size = 5  # Tamaño de la ventana de contexto
    
    # 1. Leer el dataset
    text = read_dataset(dataset_path)
    
    # 2. Tokenizar el texto
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # +1 para el token de padding
    
    # Convertir el texto a secuencia de tokens
    tokenized_text = tokenizer.texts_to_sequences([text])[0]
    
    # 3. Crear secuencias de entrenamiento usando ventana deslizante
    X, y = create_training_sequences(tokenized_text, window_size)
    
    # Mostrar información sobre los datos de entrenamiento
    print(f"Vocabulario: {len(word_index)} palabras únicas")
    print(f"Número de secuencias de entrenamiento: {len(X)}")
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    
    # Ejemplo de una secuencia de entrada y su salida esperada
    if len(X) > 0:
        example_idx = 10  # Índice de ejemplo
        if example_idx < len(X):
            context = X[example_idx]
            target = y[example_idx]
            
            # Convertir tokens a palabras para mejor visualización
            reverse_word_index = {v: k for k, v in word_index.items()}
            context_words = [reverse_word_index.get(token, "<UNK>") for token in context]
            target_word = reverse_word_index.get(target, "<UNK>")
            
            print("\nEjemplo de secuencia de entrenamiento:")
            print(f"Contexto: {context} -> {context_words}")
            print(f"Palabra objetivo: {target} -> {target_word}")
