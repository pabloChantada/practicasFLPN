import regex as re
import matplotlib.pyplot as plt

def tokenizar_espacios(texto):
    separador = " "
    return texto.split(separador)

def tokenizar_signos_puntuacion(texto):
    pattern = r"\w+|[^\w\s]|\p{So}" 
    tokens = re.findall(pattern, texto, flags=re.UNICODE)
    return tokens

def tokenizar_n_gramas(texto, n):
    palabras = tokenizar_espacios(texto)
    n_gramas = [' '.join(palabras[i:i+n]) for i in range(len(palabras)-n+1)]
    return n_gramas

def evaluar_funciones(archivo):
    with open(archivo, 'r', encoding='utf-8') as file:
        textos = file.readlines()
    
    for texto in textos:
        texto = texto.strip()
        print(f"Texto: {texto}")
        print("Tokenización por espacios:", tokenizar_espacios(texto))
        print("Tokenización por signos de puntuación:", tokenizar_signos_puntuacion(texto))
        print("Tokenización en n-gramas (n=2):", tokenizar_n_gramas(texto, n=2))
        print("\n")

def analizar_vocabulario(archivo):
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
        tokens_ngrams = tokenizar_n_gramas(texto, n=2)
        
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

# evaluar_funciones('test_sentences.txt')

# analizar_vocabulario('majesty_speeches.txt')

if __name__ == "__main__":
    pass
