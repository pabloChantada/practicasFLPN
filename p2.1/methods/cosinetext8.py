from tensorflow.keras.layers import Embedding, Dense, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from dataset_reader import create_context, read_dataset, load_target_words
from visualize import visualize_tsne_embeddings
from cosine_sim import compute_cosine_similarities, save_cosine_similarities
from model_words import WordModel
import numpy as np
import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class WordModel(Model):
    def __init__(self, vocab_size, embedding_size, window_size, **kwargs):
        """
        Inicializa el modelo de predicción de palabras dado su contexto.

        Args:
            vocab_size (int): Tamaño del vocabulario.
            embedding_size (int): Dimensión de los embeddings.
            window_size (int): Tamaño de la ventana de contexto (número de palabras a cada lado).
        """
        super(WordModel, self).__init__(**kwargs)

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
        Define el paso hacia adelante del modelo.

        Args:
            inputs (tensor): Ventana de contexto (batch_size, 2 * window_size).

        Returns:
            tensor: Predicción de la palabra objetivo (batch_size, vocab_size).
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
        Método opcional para ver el modelo antes de entrenarlo.

        Returns:
            Model: Modelo de Keras con entradas y salidas definidas.
        """
        inputs = Input(shape=(2 * self.window_size,), name="input_context")
        return Model(inputs=inputs, outputs=self.call(inputs))
    
    def get_config(self):
        """
        Devuelve la configuración del modelo, incluyendo los parámetros necesarios.
        """
        config = super(WordModel, self).get_config()  # Obtener configuración básica
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "window_size": self.window_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstruir el modelo a partir de la configuración.
        """
        return cls(
            vocab_size=config["vocab_size"],
            embedding_size=config["embedding_size"],
            window_size=config["window_size"],
            **config
        )
vocab_size = 253854  # tamaño del vocabulario
embedding_size = 300  # tamaño de los embeddings

# Crear la capa de embeddings
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size)

def load_embeddings(file_path, vocab_size, embedding_size):
    embeddings_index = {}
    
    # Cargar el archivo de embeddings
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Crear la matriz de embeddings para asignar a la capa
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Cargar los embeddings preentrenados
#embedding_matrix = load_embeddings('embeddings/word_embedding_text8.keras', vocab_size, embedding_size)


model = tf.keras.models.load_model('embeddings/word_embedding_text8.keras')

