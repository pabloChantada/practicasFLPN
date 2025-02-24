from tokenizacion import tokenizar_espacios, tokenizar_signos_puntuacion, tokenizar_n_gramas
import matplotlib.pyplot as plt
from bpe import BPE
from wordPiece import wordPiece
# from wordPiece import WordPiece

def analize_all_vocabs(file, n=2, max_vocab=3000):
    with open(file, 'r', encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    # Vocabularios acumulativos para los métodos tradicionales
    vocab_spaces = set()
    vocab_symbols = set()
    vocab_ngrams = set()
    
    vocab_size_spaces = []
    vocab_size_symbols = []
    vocab_size_ngrams = []
    vocab_sizes_bpe = []  # Lista para almacenar el tamaño del vocabulario de BPE en cada iteración
    vocab_sizes_wp = []   # Lista para almacenar el tamaño del vocabulario de WordPiece en cada iteración

    # Inicializar modelos
    bpe_model = None
    wp_model = None

    # Vamos creando el corpus a medida que aumentamos las frases
    for i in range(1, len(texts) + 1):
        sentence = texts[i - 1]
        
        # Métodos tradicionales
        vocab_spaces.update(tokenizar_espacios(sentence))
        vocab_symbols.update(tokenizar_signos_puntuacion(sentence))
        vocab_ngrams.update(tokenizar_n_gramas(sentence, n))
        
        vocab_size_spaces.append(len(vocab_spaces))
        vocab_size_symbols.append(len(vocab_symbols))
        vocab_size_ngrams.append(len(vocab_ngrams))
        
        # BPE y WordPiece: entrenamos el modelo hasta la frase actual
        if i % 250 == 0 or i == len(texts):  # Entrenar cada 500 frases o al final
            corpus = " ".join(texts[:i])
            
            # Entrenar BPE
            bpe_model = BPE(corpus, vocab_size=max_vocab)
            bpe_model.generate_vocab_with_subunits()
            current_vocab_bpe = bpe_model.get_current_vocab()
            last_bpe_value = len(current_vocab_bpe)

            # Entrenar WordPiece
            wp_model = wordPiece(corpus, vocab_size=max_vocab)
            wp_model.build_vocab()
            current_vocab_wp = wp_model.get_current_vocab()
            last_wp_value = len(current_vocab_wp)

            # Debug: Imprimir vocabularios
            print(f"Iteration {i}: BPE Vocab Size = {last_bpe_value}, WordPiece Vocab Size = {last_wp_value}")
            print("WordPiece Vocabulary:", current_vocab_wp)
        else:
            # Mantener los últimos valores conocidos
            last_bpe_value = vocab_sizes_bpe[-1] if vocab_sizes_bpe else 0
            last_wp_value = vocab_sizes_wp[-1] if vocab_sizes_wp else 0
        
        vocab_sizes_bpe.append(last_bpe_value)
        vocab_sizes_wp.append(last_wp_value)
        
    return vocab_size_spaces, vocab_size_symbols, vocab_size_ngrams, vocab_sizes_bpe, vocab_sizes_wp
if __name__ == "__main__":
    file = "majesty_speeches.txt"
    vocab_spaces, vocab_signos, vocab_ngrams, vocab_bpe, vocab_wp = analize_all_vocabs(file, n=2, max_vocab=30000)
    print(vocab_wp)
    plt.figure(figsize=(12, 8))
    plt.plot(vocab_spaces, label='Espacios')
    plt.plot(vocab_signos, label='Signos de puntuación')
    plt.plot(vocab_ngrams, label='N-gramas (n=2)')
    plt.plot(vocab_bpe, label='BPE (Vocab max=3000)')
    plt.plot(vocab_wp, label='wordpiece (Vocab max=3000)')
    plt.xlabel('Número de oraciones procesadas')
    plt.ylabel('Tamaño del vocabulario')
    plt.title('Evolución del tamaño del vocabulario')
    plt.legend()
    plt.show()
