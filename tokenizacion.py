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

texto = "Una rata pequeña persigue un ratón 🐀."
print(texto)
print(tokenizar_espacios(texto))
print(tokenizar_signos_puntuacion(texto))
print(tokenizar_n_gramas(texto, 2))