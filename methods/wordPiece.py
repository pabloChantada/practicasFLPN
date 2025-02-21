import regex as re
from collections import defaultdict, Counter
from tokenizacion import tokenizar_espacios

class wordPiece:
    def __init__(self, corpus: str, vocab_size: int):
        self.vocab = self.initialize_vocab(
            corpus
        )[0]  # Diccionario de palabras y sus divisiones

        self.corpus = corpus  # Texto completo
        self.vocab_size = vocab_size

    def preprocess_corpus(self, corpus):
        if isinstance(corpus, list):  
            corpus = [" ".join(sentence) if isinstance(sentence, list) else sentence for sentence in corpus]
            corpus = "\n".join(corpus)  # Ahora podemos unirlo en un solo string

        sentences = corpus.split("\n")  # Separamos en oraciones por saltos de línea
        return [tokenizar_espacios(sentence) for sentence in sentences if isinstance(sentence, str) and sentence.strip()]


    def initialize_vocab(self, corpus):
        vocab = {"[UNK]": 0}
        word_freq = Counter()
        subwords = {}

        sentences = self.preprocess_corpus(corpus)  # Asegura que corpus se procese correctamente

        for sentence in sentences:
            for word in sentence:
                word_freq[word] += 1
                chars = list(word)
                if chars:
                    segmented_word = [chars[0]] + ["##" + c for c in chars[1:]]
                    subwords[word] = segmented_word
                    for subword in segmented_word:
                        vocab[subword] = vocab.get(subword, 0) + 1

        return vocab, word_freq, subwords

    def get_pair_frequencies(self, subwords, word_freq):
        pair_freq = defaultdict(int)
        for word, segments in subwords.items():
            for i in range(len(segments) - 1):
                pair = (segments[i], segments[i + 1])
                pair_freq[pair] += word_freq[word]
        return pair_freq

    def merge_pair(self, pair, vocab, subwords, word_freq):
        new_subword = pair[0] + pair[1].replace("##", "")  # Concatenamos los segmentos
        vocab[new_subword] = max(vocab.get(pair[0], 0), vocab.get(pair[1], 0))  # Frecuencia correcta

        new_subwords = {}
        for word, segments in subwords.items():
            new_segments = []
            i = 0
            while i < len(segments):
                if i < len(segments) - 1 and (segments[i], segments[i + 1]) == pair:
                    new_segments.append(new_subword)
                    i += 2  # Saltamos el par ya fusionado
                else:
                    new_segments.append(segments[i])
                    i += 1
            new_subwords[word] = new_segments

        return vocab, new_subwords
        
        return vocab, new_subwords

    def build_vocab(self):
        sentences = self.preprocess_corpus(self.corpus)
        vocab, word_freq, subwords = self.initialize_vocab(sentences)

        while len(vocab) < self.vocab_size:
            pair_freq = self.get_pair_frequencies(subwords, word_freq)
            if not pair_freq:
                break  # No hay más pares para fusionar

            most_frequent_pair = max(pair_freq, key=pair_freq.get)
            vocab, subwords = self.merge_pair(most_frequent_pair, vocab, subwords, word_freq)

            if len(vocab) >= self.vocab_size:
                break  # Nos aseguramos de no pasarnos del tamaño límite

        return vocab, subwords

    def tokenize_word(self, word, vocab):
        subwords = []  # Lista donde se almacenarán las subpalabras
        remaining = word  # Parte de la palabra que aún no ha sido procesada
        
        while remaining:  # Mientras haya algo por procesar
            for i in range(len(remaining), 1, -1):  # Ahora empezamos desde longitud 2
                candidate = remaining[:i]  
                if len(subwords) > 0:
                    candidate = "##" + candidate  

                if candidate in vocab:
                    subwords.append(candidate)  
                    remaining = remaining[i:]  
                    break  
            else:
                subwords.append("[UNK]")  
                remaining = ""  
        
        return subwords

    def tokenize_sentence(self, sentence):
        words = tokenizar_espacios(sentence)
        tokens_dict = {}
        for word in words:
            tokens_dict[word] = self.tokenize_word(word, self.vocab)
        return tokens_dict
