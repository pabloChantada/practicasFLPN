from abc import abstractmethod
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
import time
import os


def load_data(file_path, task="pos"):
    """
    Carga y preprocesa datos de un archivo.
    Devuelve oraciones y etiquetas como listas de listas.
    
    Args:
        file_path: Ruta al archivo de datos
        task: 'pos' para etiquetado POS o 'ner' para reconocimiento de entidades
    """
    sentences = []
    tags = []
    
    current_sentence = []
    current_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            if line == '':  # Línea vacía indica fin de oración
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
            else:
                # Manejo de diferentes formatos de archivo
                if task == "pos" and '\t' in line:
                    word, tag = line.split('\t')
                else:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            word, tag = parts[0], parts[-1]
                        else:
                            continue  # Omitir líneas mal formadas
                    except ValueError:
                        continue  # Omitir líneas que no se pueden analizar
                
                current_sentence.append(word.lower())  # Convertir a minúsculas
                current_tags.append(tag)
    
    # Añadir la última oración si el archivo no termina con una línea vacía
    if current_sentence:
        sentences.append(current_sentence)
        tags.append(current_tags)
    
    return sentences, tags


class SequenceTagger:
    """Clase base para tareas de etiquetado de secuencia"""
    
    def __init__(self, task="pos", embedding_type="random", bidirectional=True):
        """
        Inicializa el etiquetador de secuencia.
        
        Args:
            task: 'pos' para etiquetado POS o 'ner' para reconocimiento de entidades
            embedding_type: 'random' o 'word2vec'
            bidirectional: Si se debe usar LSTM bidireccional o simple
        """
        self.task = task
        self.embedding_type = embedding_type
        self.bidirectional = bidirectional
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = {}
        self.model = Sequential()
        self.embedding_matrix = None
        self.max_len = 64
        self.embedding_dim = 200
    
    def preprocess_data(self, train_file, dev_file, test_file):
        """Carga y preprocesa todos los archivos de datos"""
        # Cargar los datos
        self.train_sentences, self.train_tags = load_data(train_file, self.task)
        self.dev_sentences, self.dev_tags = load_data(dev_file, self.task)
        self.test_sentences, self.test_tags = load_data(test_file, self.task)
        
        # Crear vocabularios de palabras y etiquetas
        all_words = set(word for sentence in self.train_sentences for word in sentence)
        
        # Para NER, queremos que las etiquetas estén ordenadas por consistencia
        if self.task == "ner":
            all_tags = set(tag for tag_seq in self.train_tags for tag in tag_seq)
            self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
        else:  # POS
            all_tags = set(tag for tag_seq in self.train_tags for tag in tag_seq)
            self.tag2idx = {tag: idx for idx, tag in enumerate(all_tags)}
        
        # Crear mapeos word2idx y tag2idx
        self.word2idx = {word: idx + 1 for idx, word in enumerate(all_words)}  # reservar 0 para padding
        self.word2idx['<PAD>'] = 0  # Añadir token de padding
        self.word2idx['<UNK>'] = len(self.word2idx)  # Añadir token desconocido
        
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        
        # Convertir oraciones y etiquetas a secuencias de índices
        def encode(sentences, tags):
            sent_idx = [[self.word2idx.get(w, self.word2idx['<UNK>']) for w in s] for s in sentences]
            tag_idx = [[self.tag2idx[t] for t in tag_seq] for tag_seq in tags]
            return sent_idx, tag_idx
        
        self.train_sentences_idx, self.train_tags_idx = encode(self.train_sentences, self.train_tags)
        self.dev_sentences_idx, self.dev_tags_idx = encode(self.dev_sentences, self.dev_tags)
        self.test_sentences_idx, self.test_tags_idx = encode(self.test_sentences, self.test_tags)
        
        # Analizar longitudes de oraciones para determinar max_len óptimo
        all_lengths = [len(s) for s in self.train_sentences] + [len(s) for s in self.dev_sentences]
        p95_length = sorted(all_lengths)[int(0.95 * len(all_lengths))]  # percentil 95

        # Establecer max_len basado en el percentil 95 (para evitar demasiado padding)
        self.max_len = min(int(p95_length), 100)  # Limitado a 100 
        print(f"Estableciendo max_len a: {self.max_len}")
    
    def load_word2vec_embeddings(self):
        """Cargar embeddings pre-entrenados de Word2Vec"""
        print("Cargando embeddings de Word2Vec...")
        word2vec = api.load('word2vec-google-news-300')
        
        # Redimensionar la dimensión de embedding para que coincida con word2vec
        self.embedding_dim = 300
        
        # Inicializar matriz de embedding con valores aleatorios
        self.embedding_matrix = np.random.uniform(-0.25, 0.25, 
                                              (len(self.word2idx), self.embedding_dim))
        
        # Llenar la matriz de embedding con valores pre-entrenados cuando estén disponibles
        for word, idx in self.word2idx.items():
            if word in word2vec:
                self.embedding_matrix[idx] = word2vec[word]
        
        print(f"Embeddings cargados para {len(self.word2idx)} palabras")
    
    def create_model(self, lstm_units=128):
        """Crear un modelo secuencial para etiquetado de secuencia"""
        self.model = Sequential()
        
        # Si se usan embeddings de word2vec
        if self.embedding_type == "word2vec" and self.embedding_matrix is not None:
            self.model.add(Embedding(
                input_dim=len(self.word2idx),
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_len,
                mask_zero=True,
                trainable=True
            ))
        else:  # Embeddings aleatorios
            self.model.add(Embedding(
                input_dim=len(self.word2idx),
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                mask_zero=True
            ))
        
        # Añadir LSTM (bidireccional o simple)
        if self.bidirectional:
            # LSTM Bidireccional
            self.model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        else:
            # LSTM simple
            self.model.add(LSTM(lstm_units, return_sequences=True))
        
        # Añadir dropout para prevenir sobreajuste
        self.model.add(Dropout(0.3))
        
        # Capa de salida
        self.model.add(TimeDistributed(Dense(len(self.tag2idx), activation='softmax')))
        
        # Compilar el modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='categorical_crossentropy', 
                        optimizer=optimizer, 
                        metrics=['accuracy'])
        
        self.model.summary()
        return self.model
    
    def prepare_data_for_training(self):
        """Preparar datos para entrenamiento y validación"""
        # Rellenar secuencias a max_len
        X_train = pad_sequences(self.train_sentences_idx, maxlen=self.max_len, padding='post')
        y_train = pad_sequences(self.train_tags_idx, maxlen=self.max_len, padding='post')
        
        # Convertir etiquetas a codificación one-hot
        y_train_onehot = np.array([to_categorical(tags, num_classes=len(self.tag2idx)) 
                                for tags in y_train])
        
        # Preparar datos de validación
        X_val = pad_sequences(self.dev_sentences_idx, maxlen=self.max_len, padding='post')
        y_val = pad_sequences(self.dev_tags_idx, maxlen=self.max_len, padding='post')
        y_val_onehot = np.array([to_categorical(tags, num_classes=len(self.tag2idx)) 
                              for tags in y_val])
        
        return X_train, y_train_onehot, X_val, y_val_onehot
    
    def train(self, epochs=30, batch_size=32):
        """Entrenar el modelo - implementación base"""
        print("Preparando datos de entrenamiento...")
        X_train, y_train_onehot, X_val, y_val_onehot = self.prepare_data_for_training()
        
        print("Entrenando modelo...")
        start_time = time.time()
        
        # Añadir early stopping para prevenir sobreajuste
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
            callbacks=[early_stopping],
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        return history
    
    def test(self):
        """Evaluar el modelo en datos de prueba"""
        print("Evaluando modelo en datos de prueba...")
        X_test = pad_sequences(self.test_sentences_idx, maxlen=self.max_len, padding='post')
        y_test = pad_sequences(self.test_tags_idx, maxlen=self.max_len, padding='post')
        y_test_onehot = np.array([to_categorical(tags, num_classes=len(self.tag2idx)) 
                                for tags in y_test])
        
        loss, accuracy = self.model.evaluate(X_test, y_test_onehot, verbose=1)
        print(f"Pérdida en prueba: {loss:.4f}, Precisión en prueba: {accuracy:.4f}")
        
        return loss, accuracy
    
    @abstractmethod
    def run(self, train_file, dev_file, test_file):
        """Pipeline completo para entrenar y evaluar el modelo"""
        pass