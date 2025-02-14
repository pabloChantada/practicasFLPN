import regex as re
from collections import defaultdict, Counter
from tokenizacion import tokenizar_espacios


def preprocess_corpus(corpus):
    # Pre-tokenizar el corpus basado en espacios
    sentences = [tokenizar_espacios(sentence) for sentence in corpus]
    return sentences

def initialize_vocab(sentences):
    vocab = {"[UNK]": 0}  # Vocabulario inicial con el token desconocido
    word_freq = Counter()  # Frecuencias de las palabras+
    subwords = {}  # Almacenar la segmentación de cada palabra
    
    # Inicializar el vocabulario con caracteres y subpalabras
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
            chars = list(word)
            segmented_word = [chars[0]] + ["##" + c for c in chars[1:]]
            subwords[word] = segmented_word
            for subword in segmented_word:
                vocab[subword] = vocab.get(subword, 0) + 1
    
    return vocab, word_freq, subwords

def get_pair_frequencies(subwords, word_freq):
    pair_freq = defaultdict(int)
    for word, segments in subwords.items():
        for i in range(len(segments) - 1):
            pair = (segments[i], segments[i + 1])
            pair_freq[pair] += word_freq[word]
    return pair_freq

def merge_pair(pair, vocab, subwords, word_freq):
    new_subword = pair[0] + pair[1].replace('##', '')
    new_vocab_freq = vocab.get(pair[0], 0) + vocab.get(pair[1], 0)  # Ajustar la frecuencia de la nueva subpalabra
    vocab[new_subword] = new_vocab_freq
    
    new_subwords = {}
    for word, segments in subwords.items():
        new_segments = []
        i = 0
        while i < len(segments):
            if i < len(segments) - 1 and (segments[i], segments[i + 1]) == pair:
                new_segments.append(new_subword)
                i += 2
            else:
                new_segments.append(segments[i])
                i += 1
        new_subwords[word] = new_segments
    
    return vocab, new_subwords

def build_vocab(corpus, max_vocab_size):
    sentences = preprocess_corpus(corpus)
    vocab, word_freq, subwords = initialize_vocab(sentences)
    
    while len(vocab) < max_vocab_size:
        pair_freq = get_pair_frequencies(subwords, word_freq)
        if not pair_freq:
            break
        most_frequent_pair = max(pair_freq, key=pair_freq.get)
        vocab, subwords = merge_pair(most_frequent_pair, vocab, subwords, word_freq)
    
    return vocab, subwords



def tokenize_word(word, vocab):
    subwords = []  # Lista donde se almacenarán las subpalabras
    remaining = word  # Parte de la palabra que aún no ha sido procesada
    
    while remaining:  # Mientras haya algo por procesar
        # Buscamos la subcadena más larga posible que exista en el vocabulario
        for i in range(len(remaining), 0, -1):  # Recorrer de la longitud de la palabra a 1
            candidate = remaining[:i]  # Tomamos una subcadena desde el inicio hasta el índice i
            # Si es una subcadena que no empieza con el primer carácter de la palabra, agregamos '##'
            if len(subwords) > 0 :
                
                candidate = "##" + candidate  # Agregar '##' al inicio de la subcadena
                
            # Si la subcadena está en el vocabulario, la usamos
            if candidate in vocab:
                subwords.append(candidate)  # Añadimos la subcadena encontrada
                remaining = remaining[i:]  # Eliminamos la parte procesada de la palabra
                break  # Ya hemos procesado una parte de la palabra, vamos a la siguiente
        else:
            # Si no encontramos ninguna coincidencia en el vocabulario, usamos [UNK]
            subwords.append("[UNK]")  # Usamos el token desconocido
            remaining = ""  # Terminamos de procesar la palabra
    
    return subwords  # Devolvemos la lista de subpalabras






def tokenize_sentence(sentence, vocab):
    words = tokenizar_espacios(sentence)
    tokens = []
    for word in words:
        tokens.extend(tokenize_word(word, vocab))
    return tokens
