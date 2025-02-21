from bpe import BPE
from wordPiece import wordPiece
class EvaluateModels:
    def __init__(self, model: str, vocab_sizes: list, train_file: str, test_file: str) -> None:
        """
        Args:
            model (str): Nombre del modelo a evaluar (BPE o WordPiece) 
            vocab_sizes (list): Lista de tamaños de vocabulario a probar.
        """
        self.model = model

        self.vocab_sizes = vocab_sizes
        self.training_lines = self.load_file(train_file)
        self.test_lines = self.load_file(test_file)

    def load_file(self, file):
        """
        Carga un archivo y devuelve una lista de líneas.

        Args:
            file (str): Ruta del archivo a cargar.
        
        Returns:
            list: Lista de líneas del archivo, sin espacios en blanco al final.
        """
        try:
            with open(file, "r") as f:
                return [line.rstrip("\n") for line in f if line.strip()]
        except FileNotFoundError:
            print("El archivo no se encontró.")
            exit(1)


    def evaluate(self):
        """
        Evalúa el modelo de tokenización sobre los conjuntos de entrenamiento y de prueba.

        Para cada tamaño de vocabulario especificado, entrena el modelo con el corpus completo de entrenamiento, 
        imprime el vocabulario final (tokens únicos) y luego tokeniza y muestra las oraciones de ambos conjuntos.
        """

        if self.training_lines is None or self.test_lines is None:
            print("Faltan archivos de entrenamiento o test.")
            return

        for vs in self.vocab_sizes:
            # Entrenamos el modelo con todo el corpus -> training_corpus
            training_corpus = " ".join(self.training_lines)

            if self.model == "BPE":
                print(f"=== Entrenamiento con vocab_size={vs} ===")
                # Entrenar el modelo BPE 
                bpe_model = BPE(training_corpus, vocab_size=vs)
                bpe_model.generate_vocab_with_subunits()
                
                # Vocabulario final (tokens únicos)
                print("Vocabulario final:")
                print(bpe_model.get_current_vocab(),"\n")
                
                # Evaluamos sobre cada oracion -> training_lines
                # Tokenizar el conjunto de entrenamiento
                print("--- Tokenización del conjunto de entrenamiento ---")
                for sentence in self.training_lines:
                    tokens_dict = bpe_model.tokenize_sentence(sentence)
                    # Reconstruir la lista de tokens en el orden original
                    token_list = []
                    for word in sentence.split():
                        token_list.extend(tokens_dict[word])
                    print(f"Input: '{sentence}' -> Tokens: {token_list}")
                print()
                
                # Tokenizar el conjunto de prueba
                print("--- Tokenización del conjunto de prueba ---")
                for sentence in self.test_lines:
                    tokens_dict = bpe_model.tokenize_sentence(sentence)
                    token_list = []
                    for word in sentence.split():
                        token_list.extend(tokens_dict[word])
                    print(f"Input: '{sentence}' -> Tokens: {token_list}")
                print("\n" + "="*40 + "\n")

            elif self.model == "wordPiece":
                model = wordPiece(training_corpus, vs)

                # Entrenamiento
                model.vocab = model.build_vocab()[0]
                print("Vocabulario entrenado:")
                print(model.vocab)
                print("\nTokenización del conjunto de entrenamiento:")
                tokens_train = model.tokenize_sentence(training_corpus)
                for word in tokens_train:
                    print(f"{word}")

                # Test
                print("\nTokenización del conjunto de prueba:")
                for sentence in self.test_lines:
                    tokens_test = model.tokenize_sentence(sentence)
                    print(f"Input: {sentence} -> Tokens: {tokens_test}")
                    for word in tokens_test:
                        print(f"{word}")
                print("-" * 40)
            else:
                print("Modelo no soportado")
                return



if __name__ == "__main__":

    print("Implementación para BPE:")
    trainer_bpe = EvaluateModels("BPE", vocab_sizes=[150], train_file="training_sentences.txt", test_file="test_sentences.txt")
    trainer_bpe.evaluate()

    print("\nImplementación para WordPiece:")
    trainer_wp = EvaluateModels("wordPiece")
    trainer_wp.evaluate()

