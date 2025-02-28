import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d  # Para interpolación

from tokenize_simple import Tokenizer
from bpe import BPE
from wordPiece import WordPiece


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
    
    # Para BPE y WordPiece, guardaremos los puntos de entrenamiento y valores
    bpe_training_points = []
    bpe_vocab_sizes = []
    wp_training_points = []
    wp_vocab_sizes = []

    # Inicializar modelos
    bpe_model = None
    wp_model = None

    # Vamos creando el corpus a medida que aumentamos las frases
    for i in range(1, len(texts) + 1):
        sentence = texts[i - 1]
        
        # Métodos tradicionales
        vocab_spaces.update(Tokenizer.tokenize_by_spaces(sentence))
        vocab_symbols.update(Tokenizer.tokenize_by_punctuation(sentence))
        vocab_ngrams.update(Tokenizer.tokenize_n_grams(sentence, n))
        
        vocab_size_spaces.append(len(vocab_spaces))
        vocab_size_symbols.append(len(vocab_symbols))
        vocab_size_ngrams.append(len(vocab_ngrams))
        
        # BPE y WordPiece: entrenamos el modelo cada 500 frases y al final 
        if (i % 500 == 0) or (i == len(texts)):
            corpus = " ".join(texts[:i])
            
            # Entrenar BPE
            bpe_model = BPE(corpus, vocab_size=max_vocab)
            bpe_model.generate_vocab_with_subunits()
            current_vocab_bpe = bpe_model.get_current_vocab()
            bpe_vocab_size = len(current_vocab_bpe)
            
            # Guardar el punto de entrenamiento y el tamaño del vocabulario
            bpe_training_points.append(i)
            bpe_vocab_sizes.append(bpe_vocab_size)

            # Entrenar WordPiece 
            wp_model = WordPiece(corpus, vocab_size=max_vocab)
            wp_model.build_vocab()
            current_vocab_wp = wp_model.get_current_vocab()
            wp_vocab_size = len(current_vocab_wp)
            wp_training_points.append(i)
            wp_vocab_sizes.append(wp_vocab_size)
            
            print(f"Iteration {i}: BPE Vocab Size = {bpe_vocab_size} | WordPiece Vocab Size = {wp_vocab_size}")

    return vocab_size_spaces, vocab_size_symbols, vocab_size_ngrams, bpe_training_points, bpe_vocab_sizes, wp_training_points, wp_vocab_sizes

def interpolate_values(x_orig, y_orig, x_target):
    """
    Interpola valores de entrenamiento para crear una curva suave
    """

    if len(x_orig) <= 1:  # No se puede interpolar con menos de 2 puntos
        return np.zeros_like(x_target)
    
    # Asegurarse de que los puntos de entrenamiento están en orden creciente
    # sorted_indices = np.argsort(x_orig)
    # x_orig = np.array(x_orig)[sorted_indices]
    # y_orig = np.array(y_orig)[sorted_indices]
    
    # Para valores antes del primer punto de entrenamiento,
    # Estimación lineal desde el origen (0,0) hasta el primer punto
    first_x, first_y = x_orig[0], y_orig[0]
    
    # Funcion de interpolacion 
    interp_func = interp1d(x_orig, y_orig, kind='cubic', bounds_error=False, 
                          fill_value=(y_orig[-1]))  # Mantener el último valor conocido
    
    # Aplicar la interpolación a todos los puntos 
    result = np.zeros_like(x_target, dtype=float)
    
    for i, x in enumerate(x_target):
        if x < first_x:
            # Interpolación lineal desde (0,0) hasta el primer punto de entrenamiento
            result[i] = (x / first_x) * first_y
        else:
            result[i] = interp_func(x)
    
    return result

if __name__ == "__main__":
    file = "majesty_speeches.txt"
    vocab_spaces, vocab_signos, vocab_ngrams, bpe_points, bpe_sizes, wp_points, wp_sizes = analize_all_vocabs(file, n=2, max_vocab=20000)
    
    # Array con los índices de las frases para la interpolacion
    x = np.arange(1, len(vocab_spaces) + 1)
    
    # Interpolar BPE
    if bpe_points and bpe_sizes:  # Verificar que hay datos para interpolar
        vocab_bpe_interpolated = interpolate_values(bpe_points, bpe_sizes, x)
    else:
        vocab_bpe_interpolated = np.zeros_like(x)
    
    # Interpolar WordPiece 
    if wp_points and wp_sizes:
        vocab_wp_interpolated = interpolate_values(wp_points, wp_sizes, x)
    else:
        vocab_wp_interpolated = np.zeros_like(x)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x, vocab_spaces, label='Espacios')
    plt.plot(x, vocab_signos, label='Signos de puntuación')
    plt.plot(x, vocab_ngrams, label='N-gramas (n=2)')
    
    plt.plot(x, vocab_bpe_interpolated, label='BPE (Vocab max=20000)', linestyle='-')
    plt.plot(x, vocab_wp_interpolated, label='WordPiece (Vocab max=20000)', linestyle='-')
    plt.scatter(wp_points, wp_sizes, color='green', s=20, alpha=0.5)  # Para indicar los puntos de entrenamiento
    
    plt.xlabel('Número de oraciones procesadas')
    plt.ylabel('Tamaño del vocabulario')
    plt.title('Evolución del tamaño del vocabulario')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
