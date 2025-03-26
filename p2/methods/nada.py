import gensim

# Cargar Word2Vec con Gensim
model = gensim.models.KeyedVectors.load_word2vec_format('embeddings/word_embedding_text8.keras', binary=True)

# Obtener el embedding para una palabra
word_vector = model['franco']