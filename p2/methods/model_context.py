import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Reshape, Dot, Dense
import numpy as np
from dataset_reader import create_training_sequences, read_dataset

# @keras.saving.register_keras_serializable()
class ContextModel(Model):
    def __init__(self, vocab_size, embedding_size=32):
        super(ContextModel, self).__init__()
        
        # Capa de embedding compartida para ambas entradas
        self.embedding_layer = Embedding(input_dim=vocab_size, 
                                         output_dim=embedding_size)
        
        # Capas para el procesamiento
        self.reshape_layer = Reshape((embedding_size, 1))
        
        # Operación dot product
        self.dot_layer = Dot(axes=1)
        
        # Capa de salida con activación sigmoid
        self.dense_layer = Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        # Separar la entrada en palabra central y palabra de contexto
        target_word, context_word = inputs[:, 0], inputs[:, 1]
        
        # Obtener embeddings
        target_embedding = self.embedding_layer(target_word)
        context_embedding = self.embedding_layer(context_word)
        
        # Reshape para poder multiplicar
        target_reshaped = self.reshape_layer(target_embedding)
        
        # Realizar el producto escalar
        dot_product = self.dot_layer([target_reshaped, context_embedding])
        
        # Pasar por la capa densa con sigmoid
        output = self.dense_layer(dot_product)
        
        return output

if __name__ == "__main__":
    # Configuración
    dataset_path = "/home/clown/2-semester/practicasFLPN/p2/materiales/game_of_thrones.txt"
    window_size = 5  # Tamaño de la ventana de contexto
    embedding_size = 64  # Dimensión de los embeddings
    
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
    
    # 4. Crear y compilar el modelo
    model = ContextModel(vocab_size, embedding_size)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # 5. Definir un conjunto de entrada para la primera ejecución del modelo
    # Esto construye el modelo antes del resumen
    dummy_input = np.array([[1, 2]])
    _ = model(dummy_input)
    
    # 6. Mostrar resumen del modelo
    model.summary()
    
    # 7. Entrenar el modelo
    model.fit(X, y, epochs=5, batch_size=64, validation_split=0.2)
    
    # 8. Guardar el modelo entrenado
    # model.save('context_prediction_model.keras')
