import regex as re
import matplotlib.pyplot as plt
from tokenizacion import tokenizar_espacios

# FASE 1
def word_to_letter(word):
    return [letter for letter in word]      

def phrase_segmentation(phrase):
    segmented_phrase = tokenizar_espacios(phrase)  # Divide la frase en palabras
    return {word: word_to_letter(word) for word in segmented_phrase}

def word_frecuency(corpus):
    return {word: corpus.count(word) for word in set(corpus)}

# FASE 2
def consecutive_subunits(corpus):
    subunits = {}
    for word in corpus:
        # Producto escalar con zip:
        # zip(hola, ola) -> (h,o),(o,l),(l,a)
        for unit in zip(word, word[1:]):  
            subunits[unit] = subunits.get(unit, 0) + 1

    print(subunits) 
    # Encontrar la frecuencia máxima
    max_frequency = max(subunits.values())

    # Obtener todas las subunidades con la frecuencia máxima
    most_frequent_units = [unit for unit, freq in subunits.items() if freq == max_frequency]
    most_frequent_units = [letters[0] + letters[1] for letters in most_frequent_units]

    return most_frequent_units

def add_to_vocab(word, vocab):
    vocab_expanded = vocab.copy()
    if isinstance(word, list):
        vocab_expanded.extend(word)  # Añadir todos los elementos de una lista
    else:
        vocab_expanded.append(word)  # Añadir un solo elemento
    return vocab_expanded

def add_to_rules(word, rules):
    # rules = dict
    # subunit = tuple(word:list): .join(word)
    # rules.insert(subunit)
    pass

if __name__ == "__main__":
    
    phrase = "Hola, es una hola muy comun en los hola"
    print('\n=====PARTE 1=====\n')
    test_frecuencia = word_frecuency(tokenizar_espacios(phrase))
    test_segmented_words = phrase_segmentation(phrase)
    vocab = tokenizar_espacios(phrase)
    print('a. Frecuencia palabras: \n',test_frecuencia, end='\n\n')
    # test_segmented_words -> las reglas de fusion ? 
    print('d. Segmentacion palabras: \n', test_segmented_words, end='\n\n')
    print('\n=====PARTE 2=====\n')
    subunits = consecutive_subunits(tokenizar_espacios(phrase)) 
    print('a. Subunidades consecutivas: \n', subunits, end='\n\n')
    print('b. Vocabularios:\n   1.Original: \n', vocab, '\n   2.Con subunidades añadidas:\n', add_to_vocab(subunits,vocab))
