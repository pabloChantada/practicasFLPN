from tagger import SequenceTagger
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
import time
import tensorflow as tf
from nervaluate import Evaluator

class NERTagger(SequenceTagger):
    def __init__(self, embedding_type="random", bidirectional=True):
        super().__init__(task="ner", embedding_type=embedding_type, bidirectional=bidirectional)
    
    def compute_class_weights(self):
        """Calcular pesos de clase para conjuntos de datos desequilibrados (útil para NER)"""
        all_tags = [tag for seq in self.train_tags_idx for tag in seq]
        class_weights = compute_class_weight(class_weight='balanced',
                                          classes=np.unique(all_tags),
                                          y=all_tags)
        return dict(enumerate(class_weights))
    
    def train(self, epochs=50, batch_size=32):
        """Entrenar el modelo con pesos de clase para manejar etiquetas NER desequilibradas"""
        print("Preparando datos de entrenamiento...")
        X_train, y_train_onehot, X_val, y_val_onehot = self.prepare_data_for_training()
        
        # Calcular y aplicar pesos de muestra para NER
        class_weights = self.compute_class_weights()
        sample_weights = []
        for seq in pad_sequences(self.train_tags_idx, maxlen=self.max_len, padding='post'):
            weights = [class_weights[label] for label in seq]
            sample_weights.append(weights)
        sample_weights = np.array(sample_weights)
        
        print("Entrenando modelo...")
        start_time = time.time()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', 
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            batch_size=batch_size,
            epochs=epochs,
            sample_weight=sample_weights,  # Usar pesos de muestra para datos desequilibrados
            callbacks=[early_stopping],
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
 
        return history
    
    def evaluate_with_nervaluate(self):
        """Evaluación especial para tarea NER usando el paquete nervaluate"""
        if Evaluator is None:
            print("Paquete nervaluate no disponible, omitiendo evaluación detallada de NER")
            return
        
        print("\nEvaluando con nervaluate...")
        
        X_test = pad_sequences(self.test_sentences_idx, maxlen=self.max_len, padding='post')
        y_true = pad_sequences(self.test_tags_idx, maxlen=self.max_len, padding='post')
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=-1)
        
        y_true_labels = []
        y_pred_labels = []
        
        for i in range(len(X_test)):
            true_seq = []
            pred_seq = []
            for j in range(self.max_len):
                if X_test[i][j] != 0:  # Excluir padding
                    true_tag = self.idx2tag[y_true[i][j]]
                    pred_tag = self.idx2tag[y_pred[i][j]]
                    true_seq.append(true_tag)
                    pred_seq.append(pred_tag)
            y_true_labels.append(true_seq)
            y_pred_labels.append(pred_seq)
        
        # Extraer tipos de entidades (eliminando los prefijos B-, I-, etc.)
        entity_tags = set(tag.split("-")[-1] for tag in self.tag2idx.keys() if tag != "O" and "-" in tag)
        
        # Evaluación
        evaluator = Evaluator(y_true_labels, y_pred_labels, tags=list(entity_tags), loader="list")
        results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
        
        print("\n[Evaluación por entidad por modo de evaluación]")
        for entity, mode_scores in results_per_tag.items():
            print(f"\nEntidad: {entity}")
            for mode in ['strict', 'exact', 'partial', 'ent_type']:
                metrics = mode_scores.get(mode, {})
                f1 = metrics.get('f1', 0)
                print(f"  [{mode.title()}] F1: {f1:.4f}")
        return results_per_tag

    def test(self):
        """Evaluar el modelo en datos de prueba con evaluación adicional específica de NER"""
        # Primero, ejecutar la evaluación normal
        loss, accuracy = super().test()
        
        # Luego ejecutar evaluación específica de NER
        results_per_tag = self.evaluate_with_nervaluate()
        
        return loss, accuracy, results_per_tag
    
    def run(self, train_file, dev_file, test_file):
        """Pipeline completo para entrenar y evaluar el modelo NER"""
        print(f"Ejecutando NER con embeddings {self.embedding_type} {'BiLSTM' if self.bidirectional else 'LSTM'}")
        
        # Preprocesar los datos
        self.preprocess_data(train_file, dev_file, test_file)
        
        # Si se usan embeddings de word2vec, cargarlos
        if self.embedding_type == "word2vec":
            self.load_word2vec_embeddings()
        
        # Crear el modelo
        self.create_model()
        
        # Entrenar el modelo
        history = self.train()
        
        # Evaluar el modelo
        loss, accuracy, results_per_tag = self.test()

        
        return history, loss, accuracy, results_per_tag


def main(embedding_type, train_file, dev_file, test_file, bidirectional=True):
    """Función principal para ejecutar etiquetado NER"""
    ner_tagger = NERTagger(embedding_type, bidirectional)
    return ner_tagger.run(train_file, dev_file, test_file)


if __name__ == "__main__":
    import argparse
    import os
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN = os.path.join(CURRENT_DIR, "datasets/MITRestaurant/train.txt")
    DEV = os.path.join(CURRENT_DIR, "datasets/MITRestaurant/dev.txt")
    TEST = os.path.join(CURRENT_DIR, "datasets/MITRestaurant/test.txt")
    
    parser = argparse.ArgumentParser(description="Entrenar un modelo para etiquetado NER.")
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
