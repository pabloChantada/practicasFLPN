from tokenizacion import tokenizar_espacios, tokenizar_signos_puntuacion, tokenizar_n_gramas
import matplotlib.pyplot as plt
from bpe import BPE
# from wordPiece import WordPiece

def analize_all_vocabs(file, n=2, max_vocab=3000):
    with open(file, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # PARA COMPROBAR SI FUNCIONA O NO
    # texts = texts[:300]

    # Vocabularios acumulativos para los metodos tradicionales 
    vocab_spaces = set()
    vocab_symbols = set()
    vocab_ngrams = set()
    
    vocab_size_spaces = []
    vocab_size_symbols = []
    vocab_size_ngrams = []
    vocab_sizes_bpe = []        # Lista para almacenar el tamaño del vocabulario de BPE en cada iteración
    vocab_sizes_wordpiece = []  # Placeholder para WordPiece (no implementado)

    # Vamos creando corpus para a medida que aumentamos las frases
    for i in range(1, len(texts) + 1):
        # Empezamos el contador en 1 para facilitar la visibilidad
        sentence = texts[i - 1]
        # Metodos tradicionales
        vocab_spaces.update(tokenizar_espacios(sentence))
        vocab_symbols.update(tokenizar_signos_puntuacion(sentence))
        vocab_ngrams.update(tokenizar_n_gramas(sentence, n))
        
        vocab_size_spaces.append(len(vocab_spaces))
        vocab_size_symbols.append(len(vocab_symbols))
        vocab_size_ngrams.append(len(vocab_ngrams))
        
        # BPE: entrenamos el modelo hasta la frase actual 

        # BPE: entrenamos el modelo hasta la frase actual 
        if i % 500 == 0:
            corpus_bpe = " ".join(texts[:i])
            bpe_model = BPE(corpus_bpe, vocab_size=max_vocab)
            bpe_model.generate_vocab_with_subunits()
            current_vocab_bpe = bpe_model.get_current_vocab()
            last_bpe_value = len(current_vocab_bpe)
        else:
            last_bpe_value = vocab_sizes_bpe[-1] if vocab_sizes_bpe else 0  # Mantiene el último valor conocido

        vocab_sizes_bpe.append(last_bpe_value)

        
    return vocab_size_spaces, vocab_size_symbols, vocab_size_ngrams, vocab_sizes_bpe, vocab_sizes_wordpiece
if __name__ == "__main__":
    file = "majesty_speeches.txt"
    vocab_spaces, vocab_signos, vocab_ngrams, vocab_bpe, _ = analize_all_vocabs(file, n=2, max_vocab=10000)

    plt.figure(figsize=(12, 8))
    plt.plot(vocab_spaces, label='Espacios')
    plt.plot(vocab_signos, label='Signos de puntuación')
    plt.plot(vocab_ngrams, label='N-gramas (n=2)')
    plt.plot(vocab_bpe, label='BPE (Vocab max=3000)')
    plt.xlabel('Número de oraciones procesadas')
    plt.ylabel('Tamaño del vocabulario')
    plt.title('Evolución del tamaño del vocabulario')
    plt.legend()
    plt.show()
