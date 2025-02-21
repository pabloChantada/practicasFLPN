import regex as re

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
