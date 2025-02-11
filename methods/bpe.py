from tokenizacion import tokenizar_espacios


class BPE:
    def __init__(self, corpus: str, vocab_size: int):
        self.vocab = self.phrase_segmentation(
            corpus
        )  # Diccionario de palabras y sus divisiones
        self.rules = {}  # Diccionario -> {(o,l): ol}
        self.rules_order = []

        self.corpus = corpus  # Texto completo
        self.vocab_size = vocab_size

    """
    FASE 1: SEGMENTACION
    """

    def word_to_letter(self, word):
        """
        Convierte una palabra a una lista de las letras que la conforman.

        Args:
            word (str): palabra a dividir
        Returns:
            list: letras de la palabra
        """
        return [letter for letter in word]

    # Deberia ser abstracto esto ?
    def phrase_segmentation(self, phrase):
        """
        Convierte un texto en un diccionario compuesto por: {palabra: letras}

        Args:
            phrase(str): frase a dividir
        Returns:
            Diccionario con las palabras y sus letras
        """
        segmented_phrase = tokenizar_espacios(phrase)  # Divide la frase en palabras
        return {word: self.word_to_letter(word) for word in segmented_phrase}

    def word_frecuency(self):
        """
        Obtiene la frecuencia de una palabra en un corpus

        Args:
            corpus(str): texto completo que contiene palabras
        Returns:
            Diccionario con las palabras del corpus y su frecuencia
        """
        # Usamos set(corpus) para evitar trabajar con duplicados y mejorar la eficiencia
        words = tokenizar_espacios(self.corpus)
        return {word: words.count(word) for word in set(words)}

    """
    FASE 1.2: SUBUNIDADES Y ACTUALIZACION DE VOCABULARIO
    """

    def consecutive_subunits(self):
        """
        Obtiene la subunidad(par de letras) con mayor frecuencia en el vocabulario

        Args:
            vocab(dict): {palabra: letras}
        Returns:
            tuple(str, list(str)): Subunidad(str -> "xy") que mas aperece en el vocabulario, unidades que la componenen)
        """
        subunits = {}

        for word in self.vocab.values():
            # Producto escalar con zip: zip(hola, ola) -> (h,o),(o,l),(l,a)
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

    def add_to_vocab(self, subunit: tuple[str, list[str]]):
        """
        Substituye las letras correspondientes a la subunidad indicada en el vocabulario.
        i.e: {hola: [H,o,l,a]} | Subunidad: "ol" -> nuevo diccionario = {hola: [H,ol,a]}

        Args:
            subunit(str,list(str)): subunidad(par de letras) a sustituir en el diccionario, se considera
            que existe en el diccionario. Lista de partes de la subunidad
        Returns:
            Nueva instancia del diccionario, donde las letras son sustituidas por la subunidad
        """

        new_vocab = {}

        for word, letters in self.vocab.items():
            new_word = []
            cnt = 0

            while cnt < len(letters):
                # Comparamos los dos tokens consecutivos con la lista de partes de la regla
                if (
                    cnt < len(letters) - 1
                    and [letters[cnt] + letters[cnt + 1]] == subunit[1]
                ):
                    new_word.append(subunit)
                    cnt += 2  # Avanzamos +2 porque estamos cogiendo dos letras
                else:
                    # Si no coinciden anadimos la letra correspondiente igual
                    new_word.append(letters[cnt])
                    cnt += 1
            # Cambiamos la palabra por la palabra fusionada
            new_vocab[word] = new_word

        self.vocab = new_vocab

    def add_to_rules(self, subunit: tuple[str, list[str]]):
        """
        Anade una regla de fusion de la forma: {(x,y): xy}

        Args:
            subunit(str, list(str)): subunidad para formar la nueva regla de fusion, y sus partes. Lista de partes de la subunidad
        """
        fusion_str, parts = subunit
        self.rules[(parts[0], parts[1])] = fusion_str
        self.rules_order.append((parts[0], parts[1]))

    def generate_vocab_with_subunits(self):
        current_size = len(self.vocab.copy())

        while current_size < self.vocab_size:
            # subunit -> (xyz, ["xy","z"])
            subunit = self.consecutive_subunits()
            if subunit is None:
                break
            self.add_to_vocab(subunit)
            self.add_to_rules(subunit)
            current_size += 1

    def tokenize_sentence(self, sentence: str):
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
    # Corpus de prueba
    corpus = "Hola caracola"
    # El vocab_size aquí es irrelevante ya que asignaremos las reglas manualmente.
    bpe = BPE(corpus, vocab_size=100)

    # # Asignamos manualmente las reglas de fusión y su orden (según el ejemplo)
    # bpe.rules = {
    #         ("o", "l"): "ol",
    #         ("ol", "a"): "ola",
    #         ("a", "c"): "ac",
    #         ("H", "ola"): "Hola"
    #         }
    # bpe.rules_order = [
    #         ("o", "l"),
    #         ("ol", "a"),
    #         ("a", "c"),
    #         ("H", "ola")
    #         ]

    # Mostramos la segmentación inicial:
    # print("Segmentación inicial:")
    # print(bpe.vocab)
    # Salida esperada:
    # {'Hola': ['H', 'o', 'l', 'a'], 'caracola': ['c', 'a', 'r', 'a', 'c', 'o', 'l', 'a']}

    # Aplicamos la tokenización (sin unir los tokens)
    # tokenized = bpe.tokenize_sentence(corpus)

    # Imprimimos el resultado final para cada palabra:
    # print("\nTokenización final (listas de subunidades):")
    # for word, tokens in tokenized.items():
    #     print(f"{word}: {tokens}")
    #
    # La salida esperada es:
    # Hola: ["Hola"]
    # caracola: ["c", "ac", "r", "a", "c", "ola"]

    file = open("/home/clown/2-semester/practicasFLPN/text/test_sentences.txt", "r")

    i = 0
    for line in file:
        line = line.rstrip("\n")
        # Se genera un BPE para la línea (en un escenario real, el entrenamiento se hace sobre un corpus)
        bpe = BPE(line, 150)
        bpe.generate_vocab_with_subunits()
        # print(f"Input: {line} -> Reglas aprendidas: {bpe.rules}")
        # Para tokenizar una nueva oración (podrías usar la misma línea o una diferente)
        tokens = bpe.tokenize_sentence(line)
        print(f"Input: {line} -> Tokens: {tokens}")
        print("\nTokenización final (listas de subunidades):")
        for word, tokens in tokens.items():
            print(f"{word}: {tokens}")
        i += 1
        if i > 3:
            break
