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
import json
# @keras.saving.register_keras_serializable()



@tf.keras.saving.register_keras_serializable()
class ContextModel(Model):
    def __init__(self, vocab_size, embedding_size):
        super(ContextModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Capas del modelo
        self.embedding_target = Embedding(input_dim=vocab_size, 
                                        output_dim=embedding_size, 
                                        name="embedding_target")
        
        self.embedding_context = Embedding(input_dim=vocab_size, 
                                         output_dim=embedding_size, 
                                         name="embedding_context")

        self.reshape_target = Reshape((embedding_size, 1))
        self.reshape_context = Reshape((embedding_size, 1))
        
        self.dot_product = Dot(axes=1)
        self.reshape_dot = Reshape((1,))
        
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        input_target, input_context = inputs

        # Obtener embeddings
        target_embedding = self.embedding_target(input_target)
        context_embedding = self.embedding_context(input_context)

        # Ajustar dimensiones
        target_embedding = self.reshape_target(target_embedding)
        context_embedding = self.reshape_context(context_embedding)

        # Producto punto
        dot_product = self.dot_product([target_embedding, context_embedding])
        dot_product = self.reshape_dot(dot_product)

        # Salida
        output = self.output_layer(dot_product)

        return output

    def get_config(self):
        """Necesario para serializar el modelo"""
        config = super(ContextModel, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Necesario para deserializar el modelo"""
        return cls(**config)

    def build_graph(self):
        """Método opcional para visualización"""
        inputs = [Input(shape=(1,), name="input_target"), 
                 Input(shape=(1,), name="input_context")]
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)
       

if __name__ == "__main__":
    # Menú para seleccionar parámetros
    
    with open("config.json") as f:
        configs = json.load(f)
    
    for config in configs:
        corpus_choice = config["corpus"]
        ventana = config["ventana"] 
        dims = config["dims"]
        batch_size = 128 
        epochs = 5 
        # Configuración según la selección del corpus
        if corpus_choice == "1":
            dataset_path = "materiales/game_of_thrones.txt"
            target_words_path = "materiales/target_words_game_of_thrones.txt"
            model_name = f"embeddings/contextModel/word_embedding_model_game_of_thrones_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/tsne_embeddings_game_of_thrones_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/cosine_similarities_game_of_thrones_win{ventana}_dim{dims}.txt"
            
        elif corpus_choice == "2":
            dataset_path = "materiales/harry_potter_and_the_philosophers_stone.txt"
            target_words_path = "materiales/target_words_harry_potter.txt"
            model_name = f"embeddings/contextModel/word_embedding_model_harry_potter_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/tsne_embeddings_harry_potter_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/cosine_similarities_harry_potter_win{ventana}_dim{dims}.txt"
            
        elif corpus_choice == "3":
            dataset_path = "materiales/the_fellowship_of_the_ring.txt"
            target_words_path = "materiales/target_words_the_fellowship_of_the_ring.txt"
            model_name = f"embeddings/contextModel/word_embedding_model_the_fellowship_of_the_ring_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/tsne_embeddings_the_fellowship_of_the_ring_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/cosine_similarities_the_fellowship_of_the_ring_win{ventana}_dim{dims}.txt"
            
        elif corpus_choice == "4":
            dataset_path = "materiales/text8.txt"
            target_words_path = "p2/materiales/target_words_text8.txt"
            model_name = f"embeddings/contextModel/word_embedding_text8_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/tsne_embeddings_text8_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/cosine_similarities_text8_win{ventana}_dim{dims}.txt"
            batch_size = 1024  
            epochs = 20       
            
        else:
            raise ValueError("Selección no válida. Introduce 1, 2, 3 ó 4.")

        
        window_size = int(ventana)
        embedding_dim = int(dims)

        

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
        model = ContextModel(vocab_size, dims)
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