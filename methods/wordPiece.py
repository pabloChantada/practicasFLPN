from collections import defaultdict, Counter
from tokenize_simple import Tokenizer


class WordPiece:
    def __init__(self, corpus: str, vocab_size: int):
        """
        Args:
            corpus (str): Texto completo que se usará para entrenar el modelo.
            vocab_size (int): Tamaño objetivo del vocabulario (número de tokens únicos).
        """
        self.vocab = self.initialize_vocab(corpus)[0]  # Diccionario de palabras y sus divisiones
        self.corpus = corpus  # Texto completo
        self.vocab_size = vocab_size

    def preprocess_corpus(self, corpus):
        """
        Preprocesa el corpus para convertirlo en un formato adecuado para el entrenamiento.

        Args:
            corpus (str or list): Corpus de texto a preprocesar.
        
        Returns:
            list: Lista de listas donde cada sublista contiene las palabras de una oración.
        """
        if isinstance(corpus, list):  
            corpus = [" ".join(sentence) if isinstance(sentence, list) else sentence for sentence in corpus]
            corpus = "\n".join(corpus) 

        sentences = corpus.split("\n")  # Separamos en oraciones por saltos de línea
        return [Tokenizer.tokenize_by_spaces(sentence) for sentence in sentences if isinstance(sentence, str) and sentence.strip()]

    def initialize_vocab(self, corpus):
        """
        Inicializa el vocabulario con tokens a nivel de carácter y almacena la frecuencia de palabras.

        Args:
            corpus (str): Corpus de texto para inicializar el vocabulario.
        
        Returns:
            tuple: Una tupla con tres elementos:
                - dict: Vocabulario inicial con tokens a nivel de carácter
                - Counter: Frecuencia de cada palabra en el corpus
                - dict: Palabras segmentadas en sus subpalabras iniciales
        """
        vocab = {"[UNK]": 0}
        word_freq = Counter()
        subwords = {}

        sentences = self.preprocess_corpus(corpus)
            
        # EXPLICAR CON COMENTARIOS ESTA FUNCION
        for sentence in sentences:
            for word in sentence:
                word_freq[word] += 1
                chars = list(word)
                if chars:
                    segmented_word = [chars[0]] + ["##" + c for c in chars[1:]]
                    subwords[word] = segmented_word
                    for subword in segmented_word:
                        vocab[subword] = vocab.get(subword, 0) + 1

        return vocab, word_freq, subwords

    def get_pair_frequencies(self, subwords, word_freq):
        """
        Calcula la frecuencia de cada par de subunidades contiguas en el vocabulario.

        Args:
            subwords (dict): Diccionario que mapea palabras a sus subunidades.
            word_freq (Counter): Contador con la frecuencia de cada palabra.
        
        Returns:
            defaultdict: Diccionario con la frecuencia de cada par de subunidades.
        """
        pair_freq = defaultdict(int)
        for word, segments in subwords.items():
            for i in range(len(segments) - 1):
                pair = (segments[i], segments[i + 1])
                pair_freq[pair] += word_freq[word]
        return pair_freq

    def merge_pair(self, pair, vocab, subwords):
        """
        Fusiona un par de subunidades en el vocabulario y actualiza las representaciones.

        Args:
            pair (tuple): Par de subunidades a fusionar (ej: ("a", "##b")).
            vocab (dict): Vocabulario actual.
            subwords (dict): Diccionario de palabras segmentadas.
        
        Returns:
            tuple: Vocabulario actualizado y diccionario de subpalabras actualizado.
        """
        # Crear la nueva subpalabra fusionando el par
        new_subword = pair[0] + pair[1].replace("##", "")

        # Añadir la nueva subpalabra al vocabulario con su frecuencia
        # La frecuencia de la nueva subpalabra es la suma de las frecuencias del par fusionado
        vocab[new_subword] = vocab.get(pair[0], 0) + vocab.get(pair[1], 0)

        # Actualizar el diccionario de subpalabras para reemplazar el par con la nueva subpalabra
        new_subwords = {}
        for word, segments in subwords.items():
            new_segments = []
            i = 0
            while i < len(segments):
                if i < len(segments) - 1 and (segments[i], segments[i + 1]) == pair:
                    new_segments.append(new_subword)
                    i += 2  # Saltamos el par ya que ha sido fusionado
                else:
                    new_segments.append(segments[i])
                    i += 1
            new_subwords[word] = new_segments

        return vocab, new_subwords    

    def build_vocab(self):
        """
        Construye el vocabulario WordPiece mediante la fusión iterativa de pares frecuentes.
        
        Returns:
            tuple: Una tupla con dos elementos:
                - dict: Vocabulario final con tokens y sus frecuencias
                - dict: Diccionario de palabras segmentadas según el vocabulario final
        """
        vocab, word_freq, subwords = self.initialize_vocab(self.preprocess_corpus(self.corpus))

        while len(vocab) < self.vocab_size:
            # Obtener las frecuencias de todos los pares
            pair_freq = self.get_pair_frequencies(subwords, word_freq)
            if not pair_freq:
                break

            if not isinstance(pair_freq, dict) or not pair_freq:
                raise ValueError("pair_freq debe ser un diccionario no vacío")

            # Encontrar el par más frecuente
            most_frequent_pair = max(pair_freq, key=lambda k: pair_freq[k])

            # Fusionar el par más frecuente y actualizar el vocabulario
            vocab, subwords = self.merge_pair(most_frequent_pair, vocab, subwords)

            # Detener si el vocabulario alcanza el tamaño deseado
            if len(vocab) >= self.vocab_size:
                break

        return vocab, subwords

    def get_current_vocab(self):
        """
        Obtiene el conjunto actual de tokens únicos en el vocabulario.

        Returns:
            set: Conjunto de tokens únicos.
        """
        return set(self.build_vocab()[0])

    def tokenize_word(self, word, vocab):
        subwords = []  # Lista donde se almacenarán las subpalabras
        remaining = word  # Parte de la palabra que aún no ha sido procesada
        
        while remaining:  # Mientras haya algo por procesar
            for i in range(len(remaining), 1, -1):  # Ahora empezamos desde longitud 2
                candidate = remaining[:i]  
                if len(subwords) > 0:
                    candidate = "##" + candidate  

                if candidate in vocab:
                    subwords.append(candidate)  
                    remaining = remaining[i:]  
                    break  
            else:
                subwords.append("[UNK]")  
                remaining = ""  
        
        return subwords

    def tokenize_sentence(self, sentence):
        """
        Tokeniza una oración completa utilizando el WordPiece.

        Args:
            sentence (str): Oración a tokenizar.
        
        Returns:
            dict: Diccionario donde las claves son las palabras originales y los valores 
                  son listas con la tokenización resultante.
        """
        words = Tokenizer.tokenize_by_spaces(sentence)
        tokens_dict = {}
        for word in words:
            tokens_dict[word] = self.tokenize_word(word, self.vocab)
        return tokens_dict


if __name__ == "__main__":
    # Leer todo el corpus de entrenamiento
    try:
        with open("training_sentences.txt", "r", encoding="utf-8") as train_file:
            training_lines = [line.rstrip("\n") for line in train_file if line.strip()]
    except FileNotFoundError:
        print("El archivo training_sentences.txt no se encontró.")
        training_lines = [
            "el rápido zorro marrón salta sobre el perro perezoso",
            "el zorro es rápido y el perro es perezoso",
            "los zorros y perros son animales"
        ]
        print("Usando corpus de ejemplo.")
        
    # Entrenamos el modelo con todo el corpus -> training_corpus
    training_corpus = " ".join(training_lines)
    # Lista de tamaños de vocabulario a probar
    vocab_sizes = [150]
    # Leer todo el conjunto de prueba
    try:
        with open("test_sentences.txt", "r", encoding="utf-8") as test_file:
            test_lines = [line.rstrip("\n") for line in test_file if line.strip()]
    except FileNotFoundError:
        print("El archivo test_sentences.txt no se encontró.")
        test_lines = [
            "el zorro veloz",
            "un perro dormido"
        ]
        print("Usando frases de prueba de ejemplo.")
    
    for vs in vocab_sizes:
        print(f"=== Entrenamiento con vocab_size={vs} ===")
        # Entrenar el modelo WordPiece
        wp_model = WordPiece(training_corpus, vocab_size=vs)
        vocab, subwords = wp_model.build_vocab()
        wp_model.vocab = vocab  # Actualizar el vocabulario en el modelo
        
        # Vocabulario final (tokens únicos)
        print("Vocabulario final:")
        print(set(vocab.keys()),"\n")
        
        # Tokenizar el conjunto de entrenamiento
        print("--- Tokenización del conjunto de entrenamiento ---")
        for sentence in training_lines:
            tokens_dict = wp_model.tokenize_sentence(sentence)
            # Reconstruir la lista de tokens en el orden original
            token_list = []
            for word in sentence.split():
                if word in tokens_dict:  # Verificar si la palabra está en el diccionario
                    token_list.extend(tokens_dict[word])
            print(f"Input: '{sentence}' -> Tokens: {token_list}")
        print()
        
        # Tokenizar el conjunto de prueba
        print("--- Tokenización del conjunto de prueba ---")
        for sentence in test_lines:
            tokens_dict = wp_model.tokenize_sentence(sentence)
            token_list = []
            for word in sentence.split():
                if word in tokens_dict:  # Verificar si la palabra está en el diccionario
                    token_list.extend(tokens_dict[word])
            print(f"Input: '{sentence}' -> Tokens: {token_list}")
        print("\n" + "="*40 + "\n")

