from tokenizacion import tokenizar_espacios, tokenizar_signos_puntuacion, tokenizar_n_gramas
import matplotlib.pyplot as plt
from bpe import BPE
from wordPiece import wordPiece
import numpy as np

def analize_all_vocabs(file, n=2, max_vocab=3000):
    with open(file, 'r',encoding="utf-8") as f:
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
    vocab_sizes_wp = []  # Placeholder para WordPiece (no implementado)

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
            corpus = " ".join(texts[:i])
            bpe_model = BPE(corpus, vocab_size=max_vocab)
            bpe_model.generate_vocab_with_subunits()
            current_vocab_bpe = bpe_model.get_current_vocab()
            last_bpe_value = len(current_vocab_bpe)

            wp_model = wordPiece(corpus, vocab_size=max_vocab)
            wp_model.build_vocab()
            current_vocab_wp = wp_model.get_current_vocab()
            last_wp_value = len(current_vocab_wp)
        else:
            last_bpe_value = vocab_sizes_bpe[-1] if vocab_sizes_bpe else 0  # Mantiene el último valor conocido
            last_wp_value = vocab_sizes_wp[-1] if vocab_sizes_wp else 0 

        vocab_sizes_bpe.append(last_bpe_value)
        vocab_sizes_wp.append(last_wp_value)
        
    return vocab_size_spaces, vocab_size_symbols, vocab_size_ngrams, vocab_sizes_bpe, vocab_sizes_wp
if __name__ == "__main__":
    file = "majesty_speeches.txt"
    vocab_spaces, vocab_signos, vocab_ngrams, vocab_bpe, vocab_wp = analize_all_vocabs(file, n=2, max_vocab=30000)


    def interpolate_values(values, total_length, step=500):
        """Interpolar valores para que tengan el mismo tamaño que el resto de las listas."""
        if len(values) > 1:
            x_existing = np.arange(0, total_length, step)  # Puntos donde se calcularon los valores
            y_existing = np.array(values)  # Tamaño del vocabulario en esos puntos
            x_interp = np.arange(total_length)  # Puntos donde queremos valores interpolados
            return np.interp(x_interp, x_existing, y_existing)  # Interpolación lineal
        return np.zeros(total_length)  # Si hay un solo punto, usamos ceros

    # Aplicamos interpolación a BPE y WordPiece
    vocab_bpe_interp = interpolate_values(vocab_bpe, len(vocab_spaces), step=500)
    vocab_wp_interp = interpolate_values(vocab_wp, len(vocab_spaces), step=500)

    # Graficamos con interpolación
    plt.figure(figsize=(12, 8))
    plt.plot(vocab_spaces, label='Espacios')
    plt.plot(vocab_signos, label='Signos de puntuación')
    plt.plot(vocab_ngrams, label='N-gramas (n=2)')
    plt.plot(vocab_bpe_interp, label='BPE (Interpolado, Vocab max=3000)', linestyle='dashed')
    plt.plot(vocab_wp_interp, label='WordPiece (Interpolado, Vocab max=3000)', linestyle='dashed')
    plt.xlabel('Número de oraciones procesadas')
    plt.ylabel('Tamaño del vocabulario')
    plt.title('Evolución del tamaño del vocabulario')
    plt.legend()
    plt.show()

