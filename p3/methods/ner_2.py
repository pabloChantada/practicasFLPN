import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from nervaluate import Evaluator
import gensim.downloader as api
import os


MAX_LEN = 100
EMBEDDING_DIM = 100

# ---------- Cargar Datos ----------
def load_data(filename):
    sentences, labels = [], []
    with open(filename, encoding='utf-8') as f:
        sentence, label_seq = [], []
        for line in f:
            if line.strip():
                word, label = line.strip().split()
                sentence.append(word)
                label_seq.append(label)
            else:
                sentences.append(sentence)
                labels.append(label_seq)
                sentence, label_seq = [], []
    return sentences, labels

from collections import Counter

def compute_class_weights(labels, label_tokenizer):
    all_labels = [label for sentence in labels for label in sentence]
    label_counts = Counter(all_labels)
    total = sum(label_counts.values())
    
    # Mapea cada etiqueta al índice correspondiente
    weights = {}
    for label, count in label_counts.items():
        idx = label_tokenizer.word_index[label]
        weights[idx] = total / (len(label_counts) * count)  # inversamente proporcional a la frecuencia
    
    return weights

def compute_sample_weights(label_sequences, class_weights):
    sample_weights = []
    for seq in label_sequences:
        weights = [class_weights.get(label, 1.0) for label in seq]
        sample_weights.append(weights)
    sample_weights = pad_sequences(sample_weights, maxlen=MAX_LEN, padding='post')
    return np.array(sample_weights, dtype=np.float32)



# ---------- Preprocesar ----------
def preprocess(sentences, labels, word_tokenizer=None, label_tokenizer=None):
    if not word_tokenizer:
        word_tokenizer = Tokenizer(lower=False, oov_token="OOV")
        word_tokenizer.fit_on_texts(sentences)
    if not label_tokenizer:
        label_tokenizer = Tokenizer(lower=False)
        label_tokenizer.fit_on_texts(labels)

    X = word_tokenizer.texts_to_sequences(sentences)
    y = label_tokenizer.texts_to_sequences(labels)

    X = pad_sequences(X, maxlen=MAX_LEN, padding='post')
    y = pad_sequences(y, maxlen=MAX_LEN, padding='post')

    # y = np.array([np.eye(len(label_tokenizer.word_index) + 1)[seq] for seq in y])

    y_label_ids = pad_sequences(label_tokenizer.texts_to_sequences(labels), maxlen=MAX_LEN, padding='post')
    y_one_hot = np.array([np.eye(len(label_tokenizer.word_index) + 1)[seq] for seq in y_label_ids])
    return X, y_one_hot, y_label_ids, word_tokenizer, label_tokenizer

# ---------- Cargar Word2Vec ----------
def load_word2vec(word_index):
    print("Cargando Word2Vec...")
    word2vec = api.load("word2vec-google-news-300")
    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec:
            embedding_matrix[i] = word2vec[word][:EMBEDDING_DIM]
    return embedding_matrix

# ---------- Modelo ----------
def build_model(vocab_size, tag_size, embedding_matrix=None):
    model = Sequential()
    if embedding_matrix is not None:
        model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                            weights=[embedding_matrix], input_length=MAX_LEN, trainable=False, mask_zero=True))
    else:
        model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN, mask_zero=True))

    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(TimeDistributed(Dense(tag_size, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------- Decodificar etiquetas ----------
def decode_predictions(predictions, label_tokenizer):
    index2label = {v: k for k, v in label_tokenizer.word_index.items()}
    index2label[0] = 'O'
    decoded = []
    for seq in predictions:
        decoded_seq = [index2label[np.argmax(token)] for token in seq]
        decoded.append(decoded_seq)

    return decoded

def remove_padding(sequences, original_sentences):
    """
    Elimina los tokens de padding en las predicciones y etiquetas reales
    según la longitud real de cada oración original.
    """
    trimmed = []
    for pred, orig in zip(sequences, original_sentences):
        trimmed.append(pred[:len(orig)])
    return trimmed

def bio_to_entities(labels):
    entities = []
    entity = None
    for idx, tag in enumerate(labels):
        if tag.startswith("B-"):
            if entity:
                entities.append(entity)
            entity = [tag[2:], idx, idx + 1]
        elif tag.startswith("I-") and entity and tag[2:] == entity[0]:
            entity[2] += 1
        else:
            if entity:
                entities.append(entity)
                entity = None
    if entity:
        entities.append(entity)
    return [tuple(e) for e in entities]



# ---------- nervaluate ----------
def evaluate_ner(y_true, y_pred):
    tags = list(set([label for sentence in y_true + y_pred for label in sentence]))
    evaluator = Evaluator(y_true, y_pred, loader="list", tags=tags)
    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
    return results



# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", choices=["random", "word2vec"], required=True)
    parser.add_argument("--task", choices=["ner", "pos"], required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()

    # Cargar y procesar datos
    train_sents, train_labels = load_data(args.train)
    dev_sents, dev_labels = load_data(args.dev)
    test_sents, test_labels = load_data(args.test)

    X_train, y_train, y_train_ids, word_tokenizer, label_tokenizer = preprocess(train_sents, train_labels)

    X_dev, y_dev, _, _, _ = preprocess(dev_sents, dev_labels, word_tokenizer, label_tokenizer)
    X_test, y_test, _, _, _ = preprocess(test_sents, test_labels, word_tokenizer, label_tokenizer)
    class_weights = compute_class_weights(train_labels, label_tokenizer)
    sample_weights = compute_sample_weights(y_train_ids, class_weights)

    # Embeddings
    embedding_matrix = None
    if args.embedding == "word2vec":
        embedding_matrix = load_word2vec(word_tokenizer.word_index)

    # Modelo
    model = build_model(len(word_tokenizer.word_index) + 1,
                        len(label_tokenizer.word_index) + 1,
                        embedding_matrix)
    

    # Entrenar con sample_weight
    model.fit(X_train, y_train, validation_data=(X_dev, y_dev), 
          sample_weight=sample_weights, epochs=5, batch_size=32)

    
    # Evaluación
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy en test: {accuracy:.4f}")

if args.task == "ner":
    y_pred = model.predict(X_test)
    decoded_preds = decode_predictions(y_pred, label_tokenizer)
    decoded_trues = decode_predictions(y_test, label_tokenizer)

    # Elimina padding usando la longitud real de las oraciones
    decoded_preds = remove_padding(decoded_preds, test_sents)
    decoded_trues = remove_padding(decoded_trues, test_sents)

    true_entities = [bio_to_entities(seq) for seq in decoded_trues]
    pred_entities = [bio_to_entities(seq) for seq in decoded_preds]


    results = evaluate_ner(decoded_trues, decoded_preds)

    # Imprimir resultados de evaluación
    print("NER Evaluation (nervaluate):")
    print(f"Strict F1-score: {results['strict']['f1']:.4f}")
    print(f"Exact F1-score: {results['exact']['f1']:.4f}")
    print(f"Partial F1-score: {results['partial']['f1']:.4f}")
    print(f"Entity Type F1-score: {results['ent_type']['f1']:.4f}")



    # print("NER Evaluation (nervaluate): F1-scores por variante:")
    # for eval_type in ['strict', 'exact', 'partial', 'ent_type']:
    #     f1_score = results[eval_type]['f1']
    #     print(f"{eval_type.upper()} F1-score: {f1_score:.4f}")


