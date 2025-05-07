# ner_model.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from nervaluate import Evaluator 
import gensim.downloader as api
import time
import os


def load_data(file_path):
    """
    Load and preprocess NER data from a file.
    Returns sentences and ner tags as lists of lists.
    """
    sentences = []
    tags = []

    current_sentence = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1]
                    current_sentence.append(word.lower())
                    current_tags.append(tag)
        if current_sentence:
            sentences.append(current_sentence)
            tags.append(current_tags)

    return sentences, tags


class NERTagger:
    def __init__(self, embedding_type="random"):
        self.embedding_type = embedding_type
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = {}
        self.model = Sequential()
        self.embedding_matrix = None
        self.max_len = 64
        self.embedding_dim = 200

    def preprocess_data(self, train_file, dev_file, test_file):
        self.train_sentences, self.train_tags = load_data(train_file)
        self.dev_sentences, self.dev_tags = load_data(dev_file)
        self.test_sentences, self.test_tags = load_data(test_file)

        all_words = set(word for sentence in self.train_sentences for word in sentence)
        all_tags = set(tag for tag_seq in self.train_tags for tag in tag_seq)

        self.word2idx = {word: idx + 1 for idx, word in enumerate(all_words)}
        self.word2idx['<PAD>'] = 0
        self.word2idx['<UNK>'] = len(self.word2idx)

        self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}

        def encode(sentences, tags):
            sent_idx = [[self.word2idx.get(w, self.word2idx['<UNK>']) for w in s] for s in sentences]
            tag_idx = [[self.tag2idx[t] for t in tag_seq] for tag_seq in tags]
            return sent_idx, tag_idx

        self.train_sentences_idx, self.train_tags_idx = encode(self.train_sentences, self.train_tags)
        self.dev_sentences_idx, self.dev_tags_idx = encode(self.dev_sentences, self.dev_tags)
        self.test_sentences_idx, self.test_tags_idx = encode(self.test_sentences, self.test_tags)

        lengths = [len(s) for s in self.train_sentences + self.dev_sentences]
        self.max_len = min(int(np.percentile(lengths, 95)), 100)
        print(f"Setting max_len to: {self.max_len}")

    def load_word2vec_embeddings(self):
        print("Loading Word2Vec embeddings...")
        word2vec = api.load('word2vec-google-news-300')
        self.embedding_dim = 300
        self.embedding_matrix = np.random.uniform(-0.25, 0.25, (len(self.word2idx), self.embedding_dim))
        for word, idx in self.word2idx.items():
            if word in word2vec:
                self.embedding_matrix[idx] = word2vec[word]
        print(f"Loaded embeddings for {len(self.word2idx)} words")

    def create_model(self, lstm_units=128):
        self.model = Sequential()
        if self.embedding_type == "word2vec" and self.embedding_matrix is not None:
            self.model.add(Embedding(input_dim=len(self.word2idx),
                                     output_dim=self.embedding_dim,
                                     weights=[self.embedding_matrix],
                                     input_length=self.max_len,
                                     mask_zero=True,
                                     trainable=True))
        else:
            self.model.add(Embedding(input_dim=len(self.word2idx),
                                     output_dim=self.embedding_dim,
                                     input_length=self.max_len,
                                     mask_zero=True))

        self.model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        self.model.add(Dropout(0.3))
        self.model.add(TimeDistributed(Dense(len(self.tag2idx), activation='softmax')))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=['accuracy'])
        self.model.summary()


    def compute_class_weights(self):
        all_tags = [tag for seq in self.train_tags_idx for tag in seq]
        class_weights = compute_class_weight(class_weight='balanced',
                                            classes=np.unique(all_tags),
                                            y=all_tags)
        return dict(enumerate(class_weights))

    def train(self, epochs=50, batch_size=32):
        X_train = pad_sequences(self.train_sentences_idx, maxlen=self.max_len, padding='post')
        y_train = pad_sequences(self.train_tags_idx, maxlen=self.max_len, padding='post')
        y_train = np.array([to_categorical(seq, num_classes=len(self.tag2idx)) for seq in y_train])

        X_val = pad_sequences(self.dev_sentences_idx, maxlen=self.max_len, padding='post')
        y_val = pad_sequences(self.dev_tags_idx, maxlen=self.max_len, padding='post')
        y_val = np.array([to_categorical(seq, num_classes=len(self.tag2idx)) for seq in y_val])

        # Compute and apply sample weights
        class_weights = self.compute_class_weights()
        sample_weights = []
        for seq in pad_sequences(self.train_tags_idx, maxlen=self.max_len, padding='post'):
            weights = [class_weights[label] for label in seq]
            sample_weights.append(weights)
        sample_weights = np.array(sample_weights)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                        patience=5,
                                                        restore_best_weights=True)

        history = self.model.fit(X_train, y_train,
                                validation_data=(X_val, y_val),
                                batch_size=batch_size,
                                epochs=epochs,
                                sample_weight=sample_weights,
                                callbacks=[early_stopping],
                                verbose=1)

        self.model.save("ner_model.h5")
        print("Model saved as ner_model.h5")

    def evaluate_with_nervaluate(self):
        print("\nEvaluating with nervaluate...")

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
                if X_test[i][j] != 0:  # Excluimos padding
                    true_tag = self.idx2tag[y_true[i][j]]
                    pred_tag = self.idx2tag[y_pred[i][j]]
                    true_seq.append(true_tag)
                    pred_seq.append(pred_tag)
            y_true_labels.append(true_seq)
            y_pred_labels.append(pred_seq)    

        # Extraer tipos de entidades (removiendo los prefijos B-, I-, etc.)
        entity_tags = set(tag.split("-")[-1] for tag in self.tag2idx.keys() if tag != "O")

        # Evaluación
        evaluator = Evaluator(y_true_labels, y_pred_labels, tags=list(entity_tags), loader="list")
        results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()

        # print(results_per_tag)


        print("\n[Per-Entity Evaluation by Evaluation Mode]")
        for entity, mode_scores in results_per_tag.items():
            print(f"\nEntity: {entity}")
            for mode in ['strict', 'exact', 'partial', 'ent_type']:
                metrics = mode_scores.get(mode, {})
                f1 = metrics.get('f1', 0)
                print(f"  [{mode.title()}] F1: {f1:.4f}")



    def test(self):
        X_test = pad_sequences(self.test_sentences_idx, maxlen=self.max_len, padding='post')
        y_test = pad_sequences(self.test_tags_idx, maxlen=self.max_len, padding='post')
        y_test = np.array([to_categorical(seq, num_classes=len(self.tag2idx)) for seq in y_test])
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        self.evaluate_with_nervaluate()  # <-- llamado aquí
        return loss, accuracy


    def run(self, train_file, dev_file, test_file):
        self.preprocess_data(train_file, dev_file, test_file)
        if self.embedding_type == "word2vec":
            self.load_word2vec_embeddings()
        self.create_model()
        self.train()
        self.test()



def main(embedding_type, train_file, dev_file, test_file):
    print(f"Running NER with {embedding_type} embeddings")
    ner_tagger = NERTagger(embedding_type)
    ner_tagger.run(train_file, dev_file, test_file)


