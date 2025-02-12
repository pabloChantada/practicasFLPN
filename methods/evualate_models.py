from bpe import BPE

class EvaluateModels:
    def __init__(self, model: str) -> None:
        self.model = model
        self.vocab_sizes = [100, 150, 200]
        self.training_corpus = self.load_file("/home/clown/2-semester/practicasFLPN/text/training_sentences.txt")
        self.test_corpus = self.load_file("/home/clown/2-semester/practicasFLPN/text/test_sentences.txt")

    def load_file(self, file):
        try:
            with open(file, "r") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: El archivo {file} no se encontró.")
            return None

    def evaluate(self):
        if self.training_corpus is None or self.test_corpus is None:
            print("Faltan archivos de entrenamiento o test.")
            return

        print("===== Fase de Evaluacion =====")
        sentences_train = [s for s in self.training_corpus.split("\n") if s.strip() != ""]
        sentences_test = [s for s in self.test_corpus.split("\n") if s.strip() != ""]
        corpus_train = "\n".join(sentences_train)

        for size in self.vocab_sizes:
            print(f"\n--- Probando {self.model} con vocab_size = {size} ---")
            if self.model == "BPE":
                model = BPE(corpus_train, size)

                # Entrenamiento
                model.generate_vocab_with_subunits()
                print("Vocabulario entrenado:")
                print(model.vocab)
                print("\nTokenización del conjunto de entrenamiento:")
                tokens_train = model.tokenize_sentence(corpus_train)
                for word, token_list in tokens_train.items():
                    print(f"{word}: {token_list}")

                # Test
                print("\nTokenización del conjunto de prueba:")
                for sentence in sentences_test:
                    tokens_test = model.tokenize_sentence(sentence)
                    print(f"Input: {sentence} -> Tokens: {tokens_test}")
                    for word, token_list in tokens_test.items():
                        print(f"{word}: {token_list}")
                print("-" * 40)

            elif self.model == "WordPiece":
                pass
                # model = WordPiece(corpus_train, size)
            else:
                print("Modelo no soportado")
                return



if __name__ == "__main__":
    print("Implementación para BPE:")
    trainer_bpe = EvaluateModels("BPE")
    trainer_bpe.evaluate()

    # print("\nImplementación para WordPiece:")
    # trainer_wp = TrainModels("WordPiece")
    # trainer_wp.train()

