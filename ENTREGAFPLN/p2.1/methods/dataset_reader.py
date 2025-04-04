import numpy as np
import random


def read_dataset(filepath):
    """
    Lee el dataset desde un archivo y devuelve el texto como una lista de palabras.
    
    Args:
        filepath (str): Ruta al archivo de texto a leer.
        
    Returns:
        list: Lista de palabras en minúsculas del texto.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().lower().split()


def load_target_words(filepath):
    """
    Carga las palabras objetivo desde un archivo y las devuelve como un conjunto.
    
    Las palabras objetivo son aquellas para las que se analizarán los embeddings.
    
    Args:
        filepath (str): Ruta al archivo que contiene las palabras objetivo.
        
    Returns:
        set: Conjunto de palabras objetivo en minúsculas.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return set(f.read().lower().split())


def create_training_pairs(tokenized_text, target_words, vocab_size, context_window=2, negSamples=True):
    """
    Genera pares de entrenamiento positivos y negativos para el modelo de contexto.
    
    Para cada palabra objetivo, extrae palabras de contexto en una ventana deslizante
    y opcionalmente genera ejemplos negativos para el entrenamiento.
    
    Args:
        tokenized_text (list): Texto tokenizado como lista de índices de palabras.
        target_words (set): Conjunto de índices de las palabras objetivo.
        vocab_size (int): Tamaño del vocabulario.
        context_window (int): Tamaño de la ventana de contexto a cada lado de la palabra.
        negSamples (bool): Si se deben generar ejemplos negativos.
        
    Returns:
        tuple: (pairs, labels) donde:
            - pairs: array numpy de pares [palabra_objetivo, palabra_contexto]
            - labels: array numpy de etiquetas (1 para relación positiva, 0 para negativa)
    """
    
    pairs, labels = [], []
    vocab_list = list(range(1, vocab_size))  # Lista de índices de palabras disponibles

    for i, word_index in enumerate(tokenized_text):
        if word_index in target_words:  # Solo si la palabra está en la lista objetivo
            # Definir la ventana de contexto
            window_start = max(i - context_window, 0)
            window_end = min(i + context_window + 1, len(tokenized_text))

            context_words = []
            # Generar pares positivos con las palabras de contexto
            for j in range(window_start, window_end):
                if i != j:  # Excluir la palabra objetivo
                    pairs.append([word_index, tokenized_text[j]])  # Relación positiva
                    labels.append(1)
                    context_words.append(tokenized_text[j])
                    
            if negSamples:
                # Generar ejemplos negativos
                for _ in range(window_start, window_end):
                    # Seleccionar una palabra aleatoria que no sea ni la objetivo ni una de contexto
                    negative_word = random.choice(vocab_list)
                    while negative_word in context_words or negative_word == word_index:  
                        negative_word = random.choice(vocab_list)

                    pairs.append([word_index, negative_word])  # Relación negativa
                    labels.append(0)

    return np.array(pairs), np.array(labels)


def create_context(tokenized_text, target_indexes, window_size=2):
    """
    Genera ventanas de contexto y palabras objetivo para el entrenamiento del modelo de contexto.
    
    Para cada ocurrencia de una palabra objetivo, extrae su ventana de contexto.
    
    Args:
        tokenized_text (list): Texto tokenizado como lista de índices de palabras.
        target_indexes (set): Conjunto de índices de las palabras objetivo.
        window_size (int): Tamaño de la ventana de contexto a cada lado.
        
    Returns:
        tuple: (X, y) donde:
            - X: array numpy de ventanas de contexto
            - y: array numpy de palabras objetivo correspondientes
    """
    X = []
    y = []

    for i in range(window_size, len(tokenized_text) - window_size):
        target = tokenized_text[i]  # Palabra objetivo

        # Solo generar contexto si la palabra objetivo está en target_indexes
        if target in target_indexes:
            # Ventana de contexto: palabras anteriores y posteriores
            context = tokenized_text[i - window_size:i] + tokenized_text[i + window_size + 1:i + window_size * 2 + 1]
            X.append(context)
            y.append(target)

    return np.array(X), np.array(y)