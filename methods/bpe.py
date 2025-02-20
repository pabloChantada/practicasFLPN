from tokenizacion import tokenizar_espacios

class BPE:
    def __init__(self, corpus: str, vocab_size: int):
        """
        Args:
            corpus (str): Texto completo que se usará para entrenar el modelo.
            vocab_size (int): Tamaño objetivo del vocabulario (número de tokens únicos).
        """
        self.vocab = self.phrase_segmentation(corpus)  # Diccionario de palabras y sus divisiones
        self.rules = {}  # Diccionario -> {(o,l): ol}
        self.rules_order = []

        self.corpus = corpus  # Texto completo
        self.vocab_size = vocab_size

    def word_to_letter(self, word):
        """
        Convierte una palabra en una lista de sus letras.

        Args:
            word (str): Palabra a dividir.
        
        Returns:
            list: Lista de letras que componen la palabra.
        """
        return [letter for letter in word]

    def phrase_segmentation(self, phrase):
        """
        Convierte un texto en un diccionario de la forma {palabra: lista_de_letras}.

        Args:
            phrase (str): Texto a dividir.
        
        Returns:
            dict: Diccionario en el que cada clave es una palabra y el valor es la lista de sus letras.
        """
        segmented_phrase = tokenizar_espacios(phrase)  # Divide la frase en palabras
        return {word: self.word_to_letter(word) for word in segmented_phrase}

    def word_frecuency(self):
        """
        Calcula la frecuencia de cada palabra del corpus almacenado en la instancia.

        Returns:
            dict: Diccionario con las palabras y su frecuencia en el corpus.
        """
        words = tokenizar_espacios(self.corpus)
        return {word: words.count(word) for word in set(words)}

    def consecutive_subunits(self):
        """
        Identifica la pareja de subunidades (dos tokens consecutivos) que ocurre con mayor frecuencia
        en el vocabulario actual (almacenado en self.vocab).

        Returns:
            tuple[str, tuple[str, str]]: Una tupla donde el primer elemento es el string resultante de fusionar
            las dos subunidades y el segundo es la tupla de las subunidades originales.
            Por ejemplo, ("ol", ("o", "l")) si "o" y "l" son la pareja más frecuente.
            Devuelve None si no hay subunidades para fusionar.
        """
        subunits = {}

        for word in self.vocab.values():
            # Producto escalar con zip: zip(hola, hola) -> (h,o),(o,l),(l,a)
            for unit in zip(word, word[1:]):
                subunits[unit] = subunits.get(unit, 0) + 1

        if not subunits:
            return None

        # Encontrar la frecuencia máxima
        max_frequency = max(subunits.values())

        # Obtener la primera subunidad con la frecuencia máxima
        for unit, freq in subunits.items():
            if freq == max_frequency:
                return ("".join(unit), unit)

    def add_to_vocab(self, subunit: tuple[str, tuple[str, str]]):
        """
        Actualiza el vocabulario fusionando todas las ocurrencias de la pareja de subunidades indicada.

        Args:
            subunit (tuple[str, tuple[str, str]]): Una tupla donde el primer elemento es el string
            fusionado (por ejemplo, "ol") y el segundo es la pareja de letras a fusionar (por ejemplo, ("o", "l")).
        """
        new_vocab = {}
        for word, letters in self.vocab.items():
            new_word = []
            cnt = 0
            while cnt < len(letters):
                # Se compara la pareja de tokens actual con la subunidad (tupla)
                if cnt < len(letters) - 1 and (letters[cnt], letters[cnt + 1]) == subunit[1]:
                    new_word.append(subunit[0])  # Se agrega la fusión, i.e: "ol"
                    cnt += 2  # Avanzamos +2 porque fusionamos dos tokens
                else:
                    new_word.append(letters[cnt])
                    cnt += 1
            new_vocab[word] = new_word
        self.vocab = new_vocab

    def add_to_rules(self, subunit: tuple[str, tuple[str, str]]):
        """
        Añade una regla de fusión al modelo.

        Args:
            subunit (tuple[str, tuple[str, str]]): Tupla que representa la fusión, por ejemplo, ("ol", ("o", "l")).
        """
        fusion_str, parts = subunit
        self.rules[(parts[0], parts[1])] = fusion_str
        self.rules_order.append((parts[0], parts[1]))

    def get_current_vocab(self):
        """
        Obtiene el conjunto actual de tokens únicos en el vocabulario.

        Returns:
            set: Conjunto de tokens únicos.
        """
        # Elementos unicos
        unique_tokens = set()
        for tokens in self.vocab.values():
            unique_tokens.update(tokens)
        return unique_tokens

    def generate_vocab_with_subunits(self):
        """
        Genera nuevas fusiones en el vocabulario (y reglas) hasta alcanzar el tamaño deseado.
        El proceso se detiene cuando el número de tokens únicos es mayor o igual a self.vocab_size.
        """
        # El limite lo establecemos como 100, 150, 200 actualmente
        while len(self.get_current_vocab()) < self.vocab_size:
            # Obtener la subunidad más frecuente para el vocabulario actual
            subunit = self.consecutive_subunits()
            if subunit is None:
                break
            self.add_to_vocab(subunit)
            self.add_to_rules(subunit)

    def tokenize_sentence(self, sentence: str):
        """
        Tokeniza una oración aplicando las reglas de fusión aprendidas.
        Las reglas se aplican en el orden en el que fueron agregadas.

        Args:
            sentence (str): Oración a tokenizar.
        
        Returns:
            dict: Diccionario donde las claves son las palabras originales y los valores son listas
                  con la tokenización resultante (subunidades fusionadas).
        """
        tokenized_words = {}
        segmentation = self.phrase_segmentation(sentence)

        for word, letters in segmentation.items():
            # Copiamos la lista para trabajar sobre ella sin modificar la original
            tokens = letters.copy()
            for rule in self.rules_order:
                fusion_val = self.rules[rule]  # Por ejemplo, si rule es ('o', 'l'), fusion_val es "ol"
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] == rule[0] and tokens[i + 1] == rule[1]:
                        # Fusionamos los dos elementos en la posición i y i+1
                        tokens[i : i + 2] = [fusion_val]
                        # No incrementamos i para verificar si la nueva unidad se puede fusionar nuevamente
                    else:
                        i += 1
            tokenized_words[word] = tokens
        return tokenized_words


if __name__ == "__main__":
    # Leer todo el corpus de entrenamiento
    try:
        with open("training_sentences.txt", "r") as train_file:
            training_lines = [line.rstrip("\n") for line in train_file if line.strip()]
    except FileNotFoundError:
        print("El archivo training_sentences.txt no se encontró.")
        exit(1)
        
    # Entrenamos el modelo con todo el corpus -> training_corpus
    training_corpus = " ".join(training_lines)
    # Lista de tamaños de vocabulario a probar
    vocab_sizes = [100, 150, 200]
    
    # Leer todo el conjunto de prueba
    try:
        with open("test_sentences.txt", "r") as test_file:
            test_lines = [line.rstrip("\n") for line in test_file if line.strip()]
    except FileNotFoundError:
        print("El archivo test_sentences.txt no se encontró.")
        exit(1)
    
    for vs in vocab_sizes:
        print(f"=== Entrenamiento con vocab_size={vs} ===")
        # Entrenar el modelo BPE 
        bpe_model = BPE(training_corpus, vocab_size=vs)
        bpe_model.generate_vocab_with_subunits()
        
        # Vocabulario final (tokens únicos)
        print("Vocabulario final:")
        print(bpe_model.get_current_vocab(),"\n")
        
        # Tokenizar el conjunto de entrenamiento
        print("--- Tokenización del conjunto de entrenamiento ---")
        for sentence in training_lines:
            tokens_dict = bpe_model.tokenize_sentence(sentence)
            # Reconstruir la lista de tokens en el orden original
            token_list = []
            for word in sentence.split():
                token_list.extend(tokens_dict[word])
            print(f"Input: '{sentence}' -> Tokens: {token_list}")
        print()
        
        # Tokenizar el conjunto de prueba
        print("--- Tokenización del conjunto de prueba ---")
        for sentence in test_lines:
            tokens_dict = bpe_model.tokenize_sentence(sentence)
            token_list = []
            for word in sentence.split():
                token_list.extend(tokens_dict[word])
            print(f"Input: '{sentence}' -> Tokens: {token_list}")
        print("\n" + "="*40 + "\n")

