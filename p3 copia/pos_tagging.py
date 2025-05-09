from tagger import SequenceTagger


class PosTagging(SequenceTagger):
    def __init__(self, embedding_type="random", bidirectional=True):
        super().__init__(task="pos", embedding_type=embedding_type, bidirectional=bidirectional)
    
    def run(self, train_file, dev_file, test_file):
        """Pipeline completo para entrenar y evaluar el modelo de etiquetado POS"""
        print(f"Ejecutando etiquetado POS con embeddings {self.embedding_type} {'BiLSTM' if self.bidirectional else 'LSTM'}")
        
        # Preprocesar los datos
        self.preprocess_data(train_file, dev_file, test_file)
        
        if self.embedding_type == "word2vec":
            self.load_word2vec_embeddings()
        
        # Crear el modelo
        self.create_model()
        
        # Entrenar el modelo
        history = self.train(epochs=30, batch_size=32)
        
        # Evaluar el modelo
        loss, accuracy = self.test()
        
        return history, loss, accuracy


def main(embedding_type, train_file, dev_file, test_file, bidirectional=True):
    """Función principal para ejecutar etiquetado POS"""
    pos_tagger = PosTagging(embedding_type, bidirectional)
    return pos_tagger.run(train_file, dev_file, test_file)


if __name__ == "__main__":
    import argparse
    import os
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN = os.path.join(CURRENT_DIR, "datasets/PartTUT/train.txt")
    DEV = os.path.join(CURRENT_DIR, "datasets/PartTUT/dev.txt")
    TEST = os.path.join(CURRENT_DIR, "datasets/PartTUT/test.txt")
    
    parser = argparse.ArgumentParser(description="Entrenar un modelo para etiquetado POS.")
    parser.add_argument("--embedding", type=str, required=False, choices=["random", "word2vec"],
                        default="random",
                        help="Tipo de inicialización de embeddings (random o word2vec).")
    parser.add_argument("--bidirectional", action="store_true", default=True,
                        help="Usar LSTM bidireccional (por defecto: True).")
    parser.add_argument("--no_bidirectional", dest="bidirectional", action="store_false",
                        help="Usar LSTM simple en lugar de bidireccional.")
    parser.add_argument("--train", type=str, required=False,
                        help=f"Ruta al conjunto de entrenamiento. Por defecto: {TRAIN}")
    parser.add_argument("--dev", type=str, required=False,
                        help=f"Ruta al conjunto de desarrollo. Por defecto: {DEV}")
    parser.add_argument("--test", type=str, required=False,
                        help=f"Ruta al conjunto de prueba. Por defecto: {TEST}")
    
    args = parser.parse_args()
    
    train_path = args.train if args.train else TRAIN
    dev_path = args.dev if args.dev else DEV
    test_path = args.test if args.test else TEST
    embedding_type = args.embedding
    
    main(embedding_type, train_path, dev_path, test_path, args.bidirectional)
