from tensorflow.keras.layers import Embedding, Dense, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from dataset_reader import create_context, read_dataset, load_target_words
from visualize import visualize_tsne_embeddings, visualize_all_tsne_embeddings
from cosine_sim import compute_cosine_similarities, save_cosine_similarities
import json

class WordModel(Model):
    """
    Este modelo toma palabras de contexto como entrada y predice la palabra objetivo
    correspondiente.
    """
    
    def __init__(self, vocab_size, embedding_size, window_size):
        """
        Args:
            vocab_size (int): Tamaño del vocabulario (número de palabras únicas).
            embedding_size (int): Dimensión de los embeddings.
            window_size (int): Tamaño de la ventana de contexto a cada lado.
        """
        super(WordModel, self).__init__()

        # Parámetros del modelo
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.window_size = window_size

        # Capa de embedding para las palabras de contexto
        self.embedding_context = Embedding(input_dim=vocab_size, 
                                           output_dim=embedding_size, 
                                           name="embedding_context")

        # Capa de promedio global para combinar los embeddings del contexto
        self.average_layer = GlobalAveragePooling1D()

        # Capa densa para predecir la palabra objetivo
        self.output_layer = Dense(vocab_size, activation='softmax', name="output_layer")

    def call(self, inputs):
        """
        Define el forward del modelo.

        Args:
            inputs (tensor): Ventana de contexto de forma (batch_size, 2*window_size).

        Returns:
            tensor: Predicción de la palabra objetivo de forma (batch_size, vocab_size).
        """
        # Obtener embeddings de las palabras de contexto
        context_embedding = self.embedding_context(inputs)  # Forma: (batch_size, 2*window_size, embedding_size)

        # Promediar los embeddings del contexto
        averaged_embedding = self.average_layer(context_embedding)  # Forma: (batch_size, embedding_size)

        # Predecir la palabra objetivo
        output = self.output_layer(averaged_embedding)  # Forma: (batch_size, vocab_size)

        return output

    def build_graph(self):
        """
        Construye y devuelve un modelo de Keras con entradas y salidas definidas.
        
        Returns:
            Model: Modelo de Keras con la arquitectura definida.
        """
        inputs = Input(shape=(2 * self.window_size,), name="input_context")
        return Model(inputs=inputs, outputs=self.call(inputs))


if __name__ == "__main__":
    # Cargar la configuración desde el archivo JSON
    with open("config.json") as f:
        configs = json.load(f)
    
    # Iterar sobre cada configuración especificada
    for config in configs:
        corpus_choice = config["corpus"]
        ventana = config["ventana"] 
        dims = config["dims"]
        batch_size = 128  	# Valor predeterminado para corpus pequeños
        epochs = 5       	# Valor predeterminado para corpus pequeños
        
        # Configuración según la selección del corpus
        if corpus_choice == "1":
            dataset_path = "materiales/game_of_thrones.txt"
            target_words_path = "materiales/target_words_game_of_thrones.txt"
            model_name = f"embeddings/wordModel/word_embedding_model_game_of_thrones_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/wordModel/tsne_embeddings_game_of_thrones_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/wordModel/cosine_similarities_game_of_thrones_win{ventana}_dim{dims}.txt"
        elif corpus_choice == "2":
            dataset_path = "materiales/harry_potter_and_the_philosophers_stone.txt"
            target_words_path = "materiales/target_words_harry_potter.txt"
            model_name = f"embeddings/wordModel/word_embedding_model_harry_potter_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/wordModel/tsne_embeddings_harry_potter_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/wordModel/cosine_similarities_harry_potter_win{ventana}_dim{dims}.txt"
        elif corpus_choice == "3":
            dataset_path = "materiales/the_fellowship_of_the_ring.txt"
            target_words_path = "materiales/target_words_the_fellowship_of_the_ring.txt"
            model_name = f"embeddings/wordModel/word_embedding_model_the_fellowship_of_the_ring_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/wordModel/tsne_embeddings_the_fellowship_of_the_ring_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/wordModel/cosine_similarities_the_fellowship_of_the_ring_win{ventana}_dim{dims}.txt"
        elif corpus_choice == "4":
            dataset_path = "materiales/text8.txt"
            target_words_path = "p2/materiales/target_words_text8.txt"
            model_name = f"embeddings/wordModel/word_embedding_text8_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/wordModel/tsne_embeddings_text8_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/wordModel/cosine_similarities_text8_win{ventana}_dim{dims}.txt"
            # Ajustar hiperparámetros para corpus más grande
            batch_size = 1024
            epochs = 20
        else:
            raise ValueError("Selección no válida. Introduce 1, 2, 3 ó 4.")

        # 1. Leer el dataset y palabras objetivo
        text = read_dataset(dataset_path)
        target_words = load_target_words(target_words_path)

        # 2. Tokenizar el texto
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
        word_index = tokenizer.word_index
        vocab_size = len(word_index) + 1  # +1 para el token de padding

        # Crear un mapeo inverso de índice a palabra
        vocab = [""] * (len(word_index) + 1)  # +1 porque los índices empiezan en 1
        for word, idx in word_index.items():
            vocab[idx] = word

        # Convertir el texto a secuencia de tokens
        tokenized_text = tokenizer.texts_to_sequences([text])[0]
        # Convertir palabras objetivo a sus índices correspondientes
        target_indexes = {word_index[word] for word in target_words if word in word_index}

        # 3. Crear secuencias de entrenamiento usando ventana deslizante
        X, y = create_context(tokenized_text, target_indexes, ventana)

        # Mostrar información sobre los datos de entrenamiento
        print(f"Vocabulario: {len(word_index)} palabras únicas")
        print(f"Número de secuencias de entrenamiento: {len(X)}")
        print(f"Forma de X: {X.shape}")
        print(f"Forma de y: {y.shape}")

        # 4. Construir y compilar el modelo
        model = WordModel(vocab_size, dims, ventana)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Mostrar resumen del modelo
        model.build_graph().summary()
        
        # 5. Visualizar embeddings iniciales (no entrenados)
        # Obtener los pesos iniciales de la capa de embedding
        embeddings = model.get_layer('embedding_context').get_weights()[0]
        
        # Visualizar embeddings iniciales de palabras objetivo
        visualize_tsne_embeddings(
            words=target_words,
            embeddings=embeddings,
            word_index=word_index,
            filename=plot_filename[:-4] + "noFitnoNeg.png"
        )
        
        # Visualizar todos los embeddings iniciales
        visualize_all_tsne_embeddings(
            embeddings=embeddings,
            word_index=word_index,
            words_to_plot=None,  # None = plotear todo el vocabulario
            words_to_label=target_words,
            filename=plot_filename[:-4] + "noFitnoNegALL.png"
        )
        
        # 6. Entrenar el modelo
        history = model.fit(
            X,  # Entrada: ventanas de contexto
            y,  # Salida: palabras objetivo
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2  # 20% para validación
        )

        # 7. Guardar el modelo entrenado
        model.save(model_name)

        # 8. Evaluar y visualizar los embeddings entrenados
        # Obtener los embeddings finales
        embeddings = model.get_layer('embedding_context').get_weights()[0]

        # Visualizar embeddings de palabras objetivo
        visualize_tsne_embeddings(
            words=target_words,
            embeddings=embeddings,
            word_index=word_index,
            filename=plot_filename[:-4] + "noNeg.png"
        )
        
        # Visualizar todos los embeddings
        visualize_all_tsne_embeddings(
            embeddings=embeddings,
            word_index=word_index,
            words_to_plot=None,
            words_to_label=target_words,
            filename=plot_filename[:-4] + "noNegALL.png"
        )
        
        # 9. Calcular similitudes de coseno entre palabras
        cosine_results = compute_cosine_similarities(target_words, word_index, embeddings)

        # 10. Imprimir y guardar similitudes de coseno
        print("\nSimilitudes de coseno:")
        for target_word, similar_words in cosine_results.items():
            print(f"Palabras más similares a '{target_word}':")
            for word, similarity in similar_words:
                print(f"  {word}: {similarity:.4f}")
            print()

        # Guardar similitudes en un archivo de texto
        save_cosine_similarities(cosine_results, cosine_filename)
        print(f"Similitudes de coseno guardadas en '{cosine_filename}'.")