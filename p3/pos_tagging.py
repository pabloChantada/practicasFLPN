import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed, Bidirectional, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import gensim.downloader as api
import time
import os


def load_data(file_path):
    """
    Load and preprocess data from a file.
    Returns sentences and tags as lists of lists.
    """
    sentences = []
    tags = []
    
    current_sentence = []
    current_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            if line == '':  # Empty line indicates end of a sentence
                if current_sentence:
                    sentences.append(current_sentence)
                    tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
            else:
                # Assumes format: word\ttag or word tag
                if '\t' in line:
                    word, tag = line.split('\t')
                else:
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            word, tag = parts[0], parts[-1]
                        else:
                            continue  # Skip malformed lines
                    except ValueError:
                        continue  # Skip lines that can't be parsed
                current_sentence.append(word.lower())  # Convert to lowercase
                current_tags.append(tag)
    
    # Add the last sentence if the file doesn't end with an empty line
    if current_sentence:
        sentences.append(current_sentence)
        tags.append(current_tags)
    
    return sentences, tags


class PosTagging:
    def __init__(self, embedding_type="random"):
        self.embedding_type = embedding_type
        self.word2idx = {}
        self.tag2idx = {}
        self.idx2tag = {}
        self.model = Sequential()
        self.embedding_matrix = None
        self.max_len = 64
        self.embedding_dim = 200  # Increased from 100 to 200
        
    def preprocess_data(self, train_file, dev_file, test_file):
        """Load and preprocess all data files"""
        # Load the data
        self.train_sentences, self.train_tags = load_data(train_file)
        self.dev_sentences, self.dev_tags = load_data(dev_file)
        self.test_sentences, self.test_tags = load_data(test_file)
        
        # Create word and tag vocabularies
        all_words = set()
        all_tags = set()
        
        for sentence in self.train_sentences:
            all_words.update(sentence)
        
        for tag_seq in self.train_tags:
            all_tags.update(tag_seq)
        
        # Create word2idx and tag2idx mappings
        self.word2idx = {word: idx + 1 for idx, word in enumerate(all_words)}  # reserve 0 for padding
        self.word2idx['<PAD>'] = 0  # Add padding token
        self.word2idx['<UNK>'] = len(self.word2idx)  # Add unknown token
        
        self.tag2idx = {tag: idx for idx, tag in enumerate(all_tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        
        # Convert sentences and tags to sequences of indices
        self.train_sentences_idx = [[self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence] 
                                  for sentence in self.train_sentences]
        self.train_tags_idx = [[self.tag2idx[tag] for tag in tag_seq] 
                            for tag_seq in self.train_tags]
        
        self.dev_sentences_idx = [[self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence] 
                                for sentence in self.dev_sentences]
        self.dev_tags_idx = [[self.tag2idx[tag] for tag in tag_seq] 
                          for tag_seq in self.dev_tags]
        
        self.test_sentences_idx = [[self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence] 
                                 for sentence in self.test_sentences]
        self.test_tags_idx = [[self.tag2idx[tag] for tag in tag_seq] 
                           for tag_seq in self.test_tags]
        
        # Print data for debugging
        print("Ejemplo de una oración de entrada:")
        if self.train_sentences:
            print(self.train_sentences[0])
            print("Etiquetas correspondientes:")
            print(self.train_tags[0])
            print("Índices de palabras:")
            print(self.train_sentences_idx[0])
            print("Índices de etiquetas:")
            print(self.train_tags_idx[0])
        
        # Verify index ranges
        if self.train_sentences_idx:
            print(f"Rango de índices de palabras: {min([min(s) for s in self.train_sentences_idx if s])} - {max([max(s) for s in self.train_sentences_idx if s])}")
            print(f"Rango de índices de etiquetas: {min([min(s) for s in self.train_tags_idx if s])} - {max([max(s) for s in self.train_tags_idx if s])}")
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        print(f"Number of tags: {len(self.tag2idx)}")
        print(f"Training samples: {len(self.train_sentences_idx)}")
        print(f"Development samples: {len(self.dev_sentences_idx)}")
        print(f"Test samples: {len(self.test_sentences_idx)}")
        
        # Analyze sentence lengths to determine optimal max_len
        all_lengths = [len(s) for s in self.train_sentences] + [len(s) for s in self.dev_sentences]
        avg_length = sum(all_lengths) / len(all_lengths)
        max_length = max(all_lengths)
        p95_length = sorted(all_lengths)[int(0.95 * len(all_lengths))]  # 95th percentile
        
        print(f"Average sentence length: {avg_length:.2f}")
        print(f"Maximum sentence length: {max_length}")
        print(f"95th percentile length: {p95_length}")
        
        # Set max_len based on the 95th percentile (to avoid padding too much)
        self.max_len = min(int(p95_length), 100)  # Cap at 100 to prevent too large models
        print(f"Setting max_len to: {self.max_len}")       

    def load_word2vec_embeddings(self):
        """Load pre-trained Word2Vec embeddings"""
        print("Loading Word2Vec embeddings...")
        word2vec = api.load('word2vec-google-news-300')
        
        # Resize embedding dimension to match word2vec
        self.embedding_dim = 300
        
        # Initialize embedding matrix with random values
        self.embedding_matrix = np.random.uniform(-0.25, 0.25, 
                                               (len(self.word2idx), self.embedding_dim))
        
        # Fill embedding matrix with pre-trained values when available
        for word, idx in self.word2idx.items():
            if word in word2vec:
                self.embedding_matrix[idx] = word2vec[word]
        
        print(f"Loaded embeddings for {len(self.word2idx)} words")

    def create_model(self, lstm_units=128):  # Increased from 64 to 128
        """Create a sequential model for POS tagging"""
        self.model = Sequential()
        
        # If using word2vec embeddings
        if self.embedding_type == "word2vec" and self.embedding_matrix is not None:
            self.model.add(Embedding(
                input_dim=len(self.word2idx),
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_len,
                mask_zero=True,
                trainable=True
            ))
        else:  # Random embeddings
            self.model.add(Embedding(
                input_dim=len(self.word2idx),
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                mask_zero=True
            ))
        
        # Bidirectional LSTM with more units
        self.model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
        # self.model.add(LSTM(lstm_units, return_sequences=True))
        
        # Add dropout to prevent overfitting
        self.model.add(Dropout(0.3))
        
        # Output layer
        self.model.add(TimeDistributed(Dense(len(self.tag2idx), activation='softmax')))
        
        # if self.embedding_type == "word2vec":
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005) # type: ignore
        # else:
        #     # Compile with custom learning rate
        #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # type: ignore
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # type: ignore

        self.model.compile(loss='categorical_crossentropy', 
                         optimizer=optimizer, 
                         metrics=['accuracy'])
        
        self.model.summary()
        return self.model
    
    def train(self, epochs=20, batch_size=32):  # Increased from 10 to 20 epochs
        """Train the model"""
        print("Preparing training data...")
        # Pad sequences to max_len
        X_train = pad_sequences(self.train_sentences_idx, maxlen=self.max_len, padding='post')
        y_train = pad_sequences(self.train_tags_idx, maxlen=self.max_len, padding='post')
        
        # Convert tags to one-hot encoding
        y_train_onehot = np.array([to_categorical(tags, num_classes=len(self.tag2idx)) 
                                 for tags in y_train])
        
        # Prepare validation data
        X_val = pad_sequences(self.dev_sentences_idx, maxlen=self.max_len, padding='post')
        y_val = pad_sequences(self.dev_tags_idx, maxlen=self.max_len, padding='post')
        y_val_onehot = np.array([to_categorical(tags, num_classes=len(self.tag2idx)) 
                               for tags in y_val])
        
        # Print shapes for debugging
        print(f"Dimensiones de X_train: {X_train.shape}")
        print(f"Dimensiones de y_train_onehot: {y_train_onehot.shape}")
        print(f"Dimensiones de X_val: {X_val.shape}")
        print(f"Dimensiones de y_val_onehot: {y_val_onehot.shape}")
        
        # Check one-hot encoding
        if y_train_onehot.size > 0:
            print(f"Ejemplo de etiqueta one-hot: {y_train_onehot[0][0]}")
            print(f"Suma de valores en ejemplo one-hot: {np.sum(y_train_onehot[0][0])}")  # Should be 1.0
        
        print("Training model...")
        start_time = time.time()
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping( # type: ignore
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
        print(f"Training completed in {training_time:.2f} seconds")
        
        
        self.model.save("pos_model.h5")
        print("Model saved as pos_model.h5")
        return history
    
    def test(self):
        """Evaluate the model on test data"""
        if self.model is None:
            print("Model is not trained yet.")
            return
        
        print("Evaluating model on test data...")
        X_test = pad_sequences(self.test_sentences_idx, maxlen=self.max_len, padding='post')
        y_test = pad_sequences(self.test_tags_idx, maxlen=self.max_len, padding='post')
        y_test_onehot = np.array([to_categorical(tags, num_classes=len(self.tag2idx)) 
                                for tags in y_test])
        
        loss, accuracy = self.model.evaluate(X_test, y_test_onehot, verbose=1)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def run(self, train_file, dev_file, test_file):
        """Complete pipeline to train and evaluate the model"""
        # Preprocess the data
        self.preprocess_data(train_file, dev_file, test_file)
        
        # If using word2vec embeddings, load them
        if self.embedding_type == "word2vec":
            self.load_word2vec_embeddings()
        
        # Create the model
        self.create_model()
        
        # Train the model
        self.train(epochs=30, batch_size=32)  # Increased from 100 to 30 epochs (with early stopping)
        
        # Test the model
        self.test()


def main(embedding_type, train_file, dev_file, test_file):
    """Main function to run POS tagging"""
    print(f"Running POS tagging with {embedding_type} embeddings")
    pos_tagger = PosTagging(embedding_type)
    pos_tagger.run(train_file, dev_file, test_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model for POS tagging.")
    parser.add_argument("--embedding", type=str, required=True, choices=["random", "word2vec"],
                        help="Type of embedding initialization (random or word2vec).")
    parser.add_argument("--train", type=str, required=True,
                        help="Path to the training set.")
    parser.add_argument("--dev", type=str, required=True,
                        help="Path to the development set.")
    parser.add_argument("--test", type=str, required=True,
                        help="Path to the test set.")
    
    args = parser.parse_args()
    
    main(args.embedding, args.train, args.dev, args.test)
