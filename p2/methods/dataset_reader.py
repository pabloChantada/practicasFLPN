from collections import defaultdict
from keras.preprocessing.text import Tokenizer
import numpy as np
import random

def read_dataset(filepath):
    """Lee el dataset y devuelve el texto como una lista de palabras."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().lower().split()

def load_target_words(filepath):
    """Carga las palabras objetivo desde un archivo y las devuelve como un conjunto."""
    with open(filepath, "r", encoding="utf-8") as f:
        return set(f.read().lower().split())

def create_training_pairs(tokenized_text, target_words, vocab_size, context_window=2, num_negative_samples=2):
    """
    Genera pares de entrenamiento positivos y negativos.

    - context_window=2 significa que se toman 2 palabras antes y 2 después.
    - Se generan ejemplos negativos para mejorar el aprendizaje.
    """
    num_negative_samples = 2*context_window
    pairs, labels = [], []
    vocab_list = list(range(1, vocab_size))  # Lista de índices de palabras disponibles

    for i, word_index in enumerate(tokenized_text):
        if word_index in target_words:  # Solo si la palabra está en la lista objetivo
            window_start = max(i - context_window, 0)
            window_end = min(i + context_window + 1, len(tokenized_text))

            context_words = []
            for j in range(window_start, window_end):
                if i != j:  
                    pairs.append([word_index, tokenized_text[j]])  # Relación positiva
                    labels.append(1)
                    context_words.append(tokenized_text[j])

            # Generar palabras negativas
            for _ in range(num_negative_samples):
                negative_word = random.choice(vocab_list)
                while negative_word in context_words or negative_word == word_index:  
                    negative_word = random.choice(vocab_list)

                pairs.append([word_index, negative_word])  # Relación negativa
                labels.append(0)

    return np.array(pairs), np.array(labels)




def create_context(tokenized_text, target_indexes, window_size=2):
    """
    Genera ventanas de contexto y palabras objetivo para las palabras objetivo específicas.

    Args:
        tokenized_text (list): Texto tokenizado (lista de índices de palabras).
        target_indexes (set): Conjunto de índices de las palabras objetivo.
        window_size (int): Tamaño de la ventana de contexto (número de palabras a cada lado).

    Returns:
        tuple: (X, y), donde X es un array de ventanas de contexto e y es un array de palabras objetivo.
    """
    X = []
    y = []

    for i in range(window_size, len(tokenized_text) - window_size):
        target = tokenized_text[i]  # Palabra objetivo

        # Solo generar contexto si la palabra objetivo está en target_indexes
        if target in target_indexes:
            # Ventana de contexto: palabras anteriores y posteriores
            context = tokenized_text[i - window_size:i] + tokenized_text[i + 1:i + window_size + 1]
            X.append(context)
            y.append(target)

    return np.array(X), np.array(y)


