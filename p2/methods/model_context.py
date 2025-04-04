import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Reshape, Dot, Dense
from dataset_reader import create_training_pairs, read_dataset, load_target_words
from visualize import visualize_tsne_embeddings, visualize_all_tsne_embeddings
from cosine_sim import compute_cosine_similarities, save_cosine_similarities
import json


@keras.saving.register_keras_serializable()
class ContextModel(Model):
    """
    Este modelo aprende a predecir palabras de contexto dada una palabra objetivo.
    """
    
    def __init__(self, vocab_size, embedding_size):
        """
        Args:
            vocab_size (int): Tamaño del vocabulario (número de palabras únicas).
            embedding_size (int): Dimensión de los embeddings.
        """
        super(ContextModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Capas del modelo
        # Embedding para palabras objetivo
        self.embedding_target = Embedding(input_dim=vocab_size, 
                                        output_dim=embedding_size, 
                                        name="embedding_target")
        
        # Embedding para palabras de contexto
        self.embedding_context = Embedding(input_dim=vocab_size, 
                                         output_dim=embedding_size, 
                                         name="embedding_context")

        # Capas para ajustar dimensiones y calcular producto punto
        self.reshape_target = Reshape((embedding_size, 1))
        self.reshape_context = Reshape((embedding_size, 1))
        self.dot_product = Dot(axes=1)
        self.reshape_dot = Reshape((1,))
        
        # Capa de salida (predicción de la relación palabra-contexto)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Define el paso forward del modelo.
        
        Args:
            inputs (list): Lista de dos tensores [palabra_objetivo, palabra_contexto].
            
        Returns:
            tensor: Predicción de la relación entre palabra objetivo y contexto (0 o 1).
        """
        input_target, input_context = inputs

        # Obtener embeddings
        target_embedding = self.embedding_target(input_target)
        context_embedding = self.embedding_context(input_context)

        # Ajustar dimensiones
        target_embedding = self.reshape_target(target_embedding)
        context_embedding = self.reshape_context(context_embedding)

        # Calcular producto punto (medida de similitud)
        dot_product = self.dot_product([target_embedding, context_embedding])
        dot_product = self.reshape_dot(dot_product)

        # Salida (probabilidad de que la palabra de contexto esté relacionada con la objetivo)
        output = self.output_layer(dot_product)

        return output

    def get_config(self):
        """
        Devuelve la configuración del modelo para serialización.
        
        Returns:
            dict: Configuración del modelo.
        """
        config = super(ContextModel, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Crea una instancia del modelo a partir de una configuración.
        
        Args:
            config (dict): Configuración del modelo.
            
        Returns:
            ContextModel: Instancia del modelo.
        """
        return cls(**config)

    def build_graph(self):
        """
        Construye y devuelve un modelo de Keras con entradas y salidas definidas.
        
        Returns:
            Model: Modelo de Keras con la arquitectura definida.
        """
        inputs = [Input(shape=(1,), name="input_target"), 
                 Input(shape=(1,), name="input_context")]
        return Model(inputs=inputs, outputs=self.call(inputs), name=self.name)
       

if __name__ == "__main__":
    # Cargar la configuración desde el archivo JSON
    with open("test.json") as f:
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
            model_name = f"embeddings/contextModel/{corpus_choice}_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/{corpus_choice}_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/{corpus_choice}_win{ventana}_dim{dims}.txt"
            
        elif corpus_choice == "2":
            dataset_path = "materiales/harry_potter_and_the_philosophers_stone.txt"
            target_words_path = "materiales/target_words_harry_potter.txt"
            model_name = f"embeddings/contextModel/{corpus_choice}_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/{corpus_choice}_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/{corpus_choice}_win{ventana}_dim{dims}.txt"
            
        elif corpus_choice == "3":
            dataset_path = "materiales/the_fellowship_of_the_ring.txt"
            target_words_path = "materiales/target_words_the_fellowship_of_the_ring.txt"
            model_name = f"embeddings/contextModel/{corpus_choice}_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/{corpus_choice}_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/{corpus_choice}_win{ventana}_dim{dims}.txt"
            
        elif corpus_choice == "4":
            dataset_path = "materiales/text8.txt"
            target_words_path = "p2/materiales/target_words_text8.txt"
            model_name = f"embeddings/contextModel/{corpus_choice}_win{ventana}_dim{dims}.keras"
            plot_filename = f"plots/contextModel/{corpus_choice}_win{ventana}_dim{dims}.png"
            cosine_filename = f"cosine/contextModel/{corpus_choice}_win{ventana}_dim{dims}.txt"
            # Ajustar hiperparámetros para corpus más grande
            batch_size = 1024  
            epochs = 20    
            
        else:
            raise ValueError("Selección no válida. Introduce 1, 2, 3 ó 4.")

        # Convertir parámetros a enteros
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
        
        # Crear un mapeo inverso de índice a palabra
        vocab = [""] * (len(word_index) + 1)  # +1 porque los índices empiezan en 1
        for word, idx in word_index.items():
            vocab[idx] = word

        # Convertir el texto a secuencia de tokens
        tokenized_text = tokenizer.texts_to_sequences([text])[0]
        # Convertir palabras objetivo a sus índices correspondientes
        target_indexes = {word_index[word] for word in target_words if word in word_index}

        # 3. Generar pares de entrenamiento (palabra objetivo, palabra contexto)
        pairs, labels = create_training_pairs(tokenized_text, target_indexes, vocab_size, window_size)

        # 4. Construir y compilar el modelo
        model = ContextModel(vocab_size, dims)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Mostrar resumen del modelo
        model.build_graph().summary()

        # Separar los pares en dos entradas: palabra objetivo y palabra de contexto
        input_targets, input_contexts = pairs[:, 0], pairs[:, 1]
        
        # Mostrar ejemplo de embedding (para depuración)
        embeddings = model.get_layer('embedding_context').get_weights()[0]
        word = "years"
        if word in word_index:
            word_idx = word_index[word]
            word_embedding = embeddings[word_idx]
            print(f"Embedding para '{word}': {word_embedding}")
        else:
            print(f"'{word}' no está en el vocabulario")
            
        # 5. Entrenar el modelo
        history = model.fit(
            [input_targets, input_contexts],  # Entradas del modelo
            labels,                           # Salida esperada
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2              # Usar 20% de los datos para validación
        )

        # 6. Guardar el modelo entrenado
        model.save(model_name)

        # 7. Obtener y visualizar los embeddings
        embeddings = model.get_layer('embedding_context').get_weights()[0]

        # Visualizar embeddings de palabras objetivo
        visualize_tsne_embeddings(
            words=target_words,
            embeddings=embeddings,
            word_index=word_index,
            filename=plot_filename[:-4] + "Neg2.png"
        )
        
        # Visualizar todos los embeddings
        visualize_all_tsne_embeddings(
            embeddings=embeddings,
            word_index=word_index,
            words_to_plot=None,
            target_words=target_words,
            filename=plot_filename[:-4] + "NegALL2.png"
        )
        
        # 8. Calcular y guardar similitudes de coseno
        cosine_results = compute_cosine_similarities(target_words, word_index, embeddings)
        save_cosine_similarities(cosine_results, cosine_filename)
        
        # 9. Imprimir información adicional
        print(f"Tamaño del vocabulario: {len(word_index)}")
        print(f"Dimensiones de embeddings: {len(embeddings)}")