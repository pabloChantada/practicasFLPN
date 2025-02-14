from tokenizacion import tokenizar_espacios, tokenizar_signos_puntuacion, tokenizar_n_gramas
import matplotlib.pyplot as plt

def analizar_vocabulario(archivo, n=2):
    with open(archivo, 'r', encoding='utf-8') as file:
        textos = file.readlines()
    
    vocab_espacios = set()
    vocab_signos = set()
    vocab_ngrams = set()
    
    vocab_size_espacios = []
    vocab_size_signos = []
    vocab_size_ngrams = []
    
    for texto in textos:
        texto = texto.strip()
        
        tokens_espacios = tokenizar_espacios(texto)
        tokens_signos = tokenizar_signos_puntuacion(texto)
        tokens_ngrams = tokenizar_n_gramas(texto, n)
        
        vocab_espacios.update(tokens_espacios)
        vocab_signos.update(tokens_signos)
        vocab_ngrams.update(tokens_ngrams)
        
        vocab_size_espacios.append(len(vocab_espacios))
        vocab_size_signos.append(len(vocab_signos))
        vocab_size_ngrams.append(len(vocab_ngrams))
    
    plt.figure(figsize=(10, 6))
    plt.plot(vocab_size_espacios, label='Espacios')
    plt.plot(vocab_size_signos, label='Signos de puntuación')
    plt.plot(vocab_size_ngrams, label='N-gramas (n=2)')
    plt.xlabel('Número de oraciones procesadas')
    plt.ylabel('Tamaño del vocabulario')
    plt.title('Evolución del tamaño del vocabulario')
    plt.legend()
    plt.show()