import tensorflow as tf
import keras
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Reshape, Dot, Dense
import numpy as np
from dataset_reader import create_training_pairs, read_dataset, load_target_words
from visualize import visualize_tsne_embeddings
from cosine_sim import compute_cosine_similarities, save_cosine_similarities
# @keras.saving.register_keras_serializable()


class ContextModel(Model):
    def __init__(self, vocab_size, embedding_size):
        super(ContextModel, self).__init__()

        # Parámetros del modelo
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Capa de embedding para la palabra objetivo
        self.embedding_target = Embedding(input_dim=vocab_size, 
                                          output_dim=embedding_size, 
                                          name="embedding_target")
        
        # Capa de embedding para la palabra de contexto
        self.embedding_context = Embedding(input_dim=vocab_size, 
                                           output_dim=embedding_size, 
                                           name="embedding_context")

        # Capa densa para generar la probabilidad de relación entre palabras
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        input_target, input_context = inputs  # Separar las entradas

        # Obtener embeddings
        target_embedding = self.embedding_target(input_target)
        context_embedding = self.embedding_context(input_context)

        # Ajustar las dimensiones
        target_embedding = Reshape((self.embedding_size, 1))(target_embedding)
        context_embedding = Reshape((self.embedding_size, 1))(context_embedding)

        # Producto punto entre los embeddings
        dot_product = Dot(axes=1)([target_embedding, context_embedding])
        dot_product = Reshape((1,))(dot_product)

        # Paso por la capa de salida
        output = self.output_layer(dot_product)

        return output

    def build_graph(self):
        """Método opcional para ver el modelo antes de entrenarlo."""
        inputs = [Input(shape=(1,), name="input_target"), Input(shape=(1,), name="input_context")]
        return Model(inputs=inputs, outputs=self.call(inputs))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE



if __name__ == "__main__":
    # Menú para seleccionar el corpus
    print("Selecciona el corpus:")
    print("1. Game of Thrones")
    print("2. Harry Potter")
    print("3. The Fellowship of the Ring")
    corpus_choice = input("Introduce el número correspondiente al corpus: ")

    # Configuración según la selección del corpus
    if corpus_choice == "1":
        dataset_path = "materiales/game_of_thrones.txt"
        target_words_path = "materiales/target_words_game_of_thrones.txt"
        model_name = "embeddings/contextModel/word_embedding_model_game_of_thrones.keras"
        plot_filename = "plots/contextModel/tsne_embeddings_game_of_thrones.png"
        cosine_filename = "cosine/contextModel/cosine_similarities_game_of_thrones.txt"
    elif corpus_choice == "2":
        dataset_path = "materiales/harry_potter_and_the_philosophers_stone.txt"
        target_words_path = "materiales/target_words_harry_potter.txt"
        model_name = "embeddings/contextModel/word_embedding_model_harry_potter.keras"
        plot_filename = "plots/contextModel/tsne_embeddings_harry_potter.png"
        cosine_filename = "cosine/contextModel/cosine_similarities_harry_potter.txt"
    elif corpus_choice == "3":
        dataset_path = "materiales/the_fellowship_of_the_ring.txt"
        target_words_path = "materiales/target_words_the_fellowship_of_the_ring.txt"
        model_name = "embeddings/contextModel/word_embedding_model_the_fellowship_of_the_ring.keras"
        plot_filename = "plots/contextModel/tsne_embeddings_the_fellowship_of_the_ring.png"
        cosine_filename = "cosine/contextModel/cosine_similarities_the_fellowship_of_the_ring.txt"
    else:
        raise ValueError("Selección no válida. Introduce 1, 2 ó 3.")

    # Configuración común
    window_size = 2  # Tamaño de la ventana de contexto (2 palabras antes y 2 después)
    embedding_size = 100  # Dimensión de los embeddings

    # 1. Leer el dataset y palabras objetivo
    text = read_dataset(dataset_path)
    target_words = load_target_words(target_words_path)

    # 2. Tokenizar el texto
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # +1 para el token de padding

    # Convertir el texto a secuencia de tokens
    tokenized_text = tokenizer.texts_to_sequences([text])[0]
    target_indexes = {word_index[word] for word in target_words if word in word_index}  # Convertir target a índices

    # 3. Crear secuencias de entrenamiento usando ventana deslizante
    pairs, labels = create_training_pairs(tokenized_text, target_indexes, vocab_size, window_size)

    # Mostrar información sobre los datos de entrenamiento
    print(f"Vocabulario: {len(word_index)} palabras únicas")
    print(f"Número de secuencias de entrenamiento: {len(pairs)}")
    print(f"Forma de X: {pairs.shape}")
    print(f"Forma de y: {labels.shape}")

    # Construir y compilar el modelo
    model = ContextModel(vocab_size, embedding_size)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Mostrar resumen del modelo
    model.build_graph().summary()

    # Separar los pares en dos entradas: palabra objetivo y palabra de contexto
    input_targets, input_contexts = pairs[:, 0], pairs[:, 1]

    # Definir hiperparámetros de entrenamiento
    batch_size = 64
    epochs = 5  # Puedes aumentar si quieres mejor rendimiento

    # Entrenar el modelo
    history = model.fit(
        [input_targets, input_contexts],  # Entradas del modelo
        labels,  # Salida esperada
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2  # Usar 20% de los datos para validación
    )

    # Guardar el modelo entrenado
    model.save(model_name)

    # Obtener los embeddings entrenados
    embeddings = model.get_layer('embedding_context').get_weights()[0]

    # Visualizar los embeddings de las palabras objetivo
    visualize_tsne_embeddings(
        words=target_words,  # Lista de palabras objetivo
        embeddings=embeddings,  # Embeddings entrenados
        word_index=word_index,  # Diccionario de palabras a índices
        filename=plot_filename  # Guardar la visualización en un archivo
    )

    # Calcular similitudes de coseno
    cosine_results = compute_cosine_similarities(target_words, word_index, embeddings)

    # Imprimir similitudes de coseno
    print("\nSimilitudes de coseno:")
    for target_word, similar_words in cosine_results.items():
        print(f"Palabras más similares a '{target_word}':")
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
        print()

   # Crear directorio si no existe
    save_cosine_similarities(cosine_results, cosine_filename)
    print(f"Similitudes de coseno guardadas en '{cosine_filename}'.")