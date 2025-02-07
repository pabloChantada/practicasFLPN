import regex as re
import matplotlib.pyplot as plt
from tokenizacion import tokenizar_espacios

# Diccionario para almacenar las reglas de fusión
def apply_merge_rule(corpus, merge_rules):
    new_corpus = []
    for word in corpus:
        for pair, merged in merge_rules.items():
            word = word.replace(" ".join(pair), merged)  # Reemplazo con la versión fusionada
        new_corpus.append(word)
    return new_corpus

# FASE 1
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
        for unit in zip(word, word[1:]):  
            subunits[unit] = subunits.get(unit, 0) + 1
    return subunits

# Función que implementa el algoritmo BPE
def bpe_algorithm(corpus, num_merges):
    merge_rules = {}  # Diccionario para almacenar reglas de fusión
    corpus = [" ".join(word) for word in corpus]  # Espaciado entre caracteres para diferenciar unidades
    
    for _ in range(num_merges):
        # Contar la frecuencia de los pares de caracteres
        subunits = consecutive_subunits(corpus)
        
        if not subunits:
            break  # No hay más pares que fusionar
        
        # Encontrar el par más frecuente

        # MODIFICAR CON EL STRING DE ABAJO
        most_frequent = max(subunits, key=subunits.get)
        merged_token = "".join(most_frequent)
        
        # Añadir la regla de fusión al diccionario
        merge_rules[most_frequent] = merged_token
        
        # Aplicar la fusión al corpus
        corpus = apply_merge_rule(corpus, merge_rules)
    
    return corpus, merge_rules

'''
    # print(subunits) 
    # Encontrar la frecuencia máxima
    max_frequency = max(subunits.values())

    # Obtener todas las subunidades con la frecuencia máxima
    most_frequent_units = [unit for unit, freq in subunits.items() if freq == max_frequency]
    most_frequent_units = [letters[0] + letters[1] for letters in most_frequent_units]
'''

if __name__ == "__main__":
    
    phrase = "Hola, es una hola muy comun en los hola"
    print('\n=====PARTE 1=====\n')
    test_frecuencia = word_frecuency(tokenizar_espacios(phrase))
    test_segmented_words = phrase_segmentation(phrase)

    print('b. Frecuencias: \n',test_frecuencia, end='\n\n')
    print('d. Segmentacion palabras: \n', test_segmented_words, end='\n\n')
    print('\n=====PARTE 2=====\n')
    print('a. Subunidades consecutivas: \n', consecutive_subunits(tokenizar_espacios(phrase)), end='\n\n')
    print()
