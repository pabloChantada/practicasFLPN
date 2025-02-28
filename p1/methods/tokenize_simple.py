import regex as re

class Tokenizer:
    
    @staticmethod
    def tokenize_by_spaces(text):
        """
        Tokeniza el texto dividiéndolo por espacios.

        Args:
            text (str): El texto a tokenizar.

        Returns:
            list: Lista de tokens obtenidos al dividir el texto por espacios.
        """
        separator = " "
        return text.split(separator)

    @staticmethod
    def tokenize_by_punctuation(text):
        """
        Tokeniza el texto teniendo en cuenta los signos de puntuación.

        Args:
            text (str): El texto a tokenizar.

        Returns:
            list: Lista de tokens obtenidos al dividir el texto en palabras y signos de puntuación.
        """
        pattern = r"\w+|[^\w\s]|\p{So}"
        tokens = re.findall(pattern, text, flags=re.UNICODE)
        return tokens

    @staticmethod
    def tokenize_n_grams(text, n):
        """
        Tokeniza el texto en n-gramas.

        Args:
            text (str): El texto a tokenizar.
            n (int): El tamaño de los n-gramas.

        Returns:
            list: Lista de n-gramas obtenidos del texto.
        """
        words = Tokenizer.tokenize_by_spaces(text)
        n_grams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        return n_grams
    
    @staticmethod
    def evaluate_functions(text):
        """
        Evalúa y muestra los resultados de las diferentes estrategias de tokenización.
        """
        print(f"Texto: {text}")
        print("Tokenización por espacios:", Tokenizer.tokenize_by_spaces(text))
        print("Tokenización por signos de puntuación:", Tokenizer.tokenize_by_punctuation(text))
        print("Tokenización en n-gramas (n=2):", Tokenizer.tokenize_n_grams(text, n=2))
        print("\n")

# Ejemplo de uso
if __name__ == "__main__":
    # Leer el archivo de texto
    try:
        with open("test_sentences.txt", "r", encoding="utf-8") as file:
            texts = file.readlines()
    except FileNotFoundError:
        print("El archivo example_text.txt no se encontró.")
        exit(1)

    # Evaluar cada texto con las funciones de tokenización
    for text in texts:
        text = text.strip()
        Tokenizer.evaluate_functions(text)
