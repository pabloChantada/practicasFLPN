import os
import argparse
import named_entity as ner
import pos_tagging as pos

# Ejemplos de ejecución:
#
# Para POS tagging con embeddings aleatorios:
# python main.py --task pos --embedding random --train /p3/datasets/PartTUT/train.txt --dev /p3/datasets/PartTUT/dev.txt --test /p3/datasets/PartTUT/test.txt
#
# Para POS tagging con embeddings word2vec:
# python main.py --task pos --embedding word2vec --train /p3/datasets/PartTUT/train.txt --dev /p3/datasets/PartTUT/dev.txt --test /p3/datasets/PartTUT/test.txt
#
# Para NER con embeddings aleatorios:
# python main.py --task ner --embedding random --train /p3/datasets/MITRestaurant/train.txt --dev /p3/datasets/MITRestaurant/dev.txt --test /p3/datasets/MITRestaurant/test.txt
#
# Para NER con embeddings word2vec:
# python main.py --task ner --embedding word2vec --train /p3/datasets/MITRestaurant/train.txt --dev /p3/datasets/MITRestaurant/dev.txt --test /p3/datasets/MITRestaurant/test.txt
#
# Para modelos sin LSTM bidireccional añadir --no_bidirectional:
# python main.py --task pos --embedding word2vec --no_bidirectional

if __name__ == "__main__":
    # Configuración de los argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Entrenar un modelo para etiquetado NER o POS.")
    parser.add_argument("--task", type=str, required=True, choices=["pos", "ner"],
                        help="Tipo de tarea a ejecutar (POS o reconocimiento de entidades)")
    parser.add_argument("--embedding", type=str, required=False, choices=["random", "word2vec"],
                        default="random",
                        help="Tipo de inicialización de embeddings (random o word2vec)")
    parser.add_argument("--bidirectional", action="store_true", default=True,
                        help="Usar LSTM bidireccional (por defecto: True).")
    parser.add_argument("--no_bidirectional", dest="bidirectional", action="store_false",
                        help="Usar LSTM simple en lugar de bidireccional.")
    parser.add_argument("--train", type=str, required=False,
                        help="Ruta al conjunto de entrenamiento")
    parser.add_argument("--dev", type=str, required=False,
                        help="Ruta al conjunto de desarrollo")
    parser.add_argument("--test", type=str, required=False,
                        help="Ruta al conjunto de prueba")
    
    args = parser.parse_args()

    # Definir rutas predeterminadas según la tarea seleccionada
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if args.task == "pos":
        TRAIN = os.path.join(CURRENT_DIR, "datasets/PartTUT/train.txt")
        DEV = os.path.join(CURRENT_DIR, "datasets/PartTUT/dev.txt")
        TEST = os.path.join(CURRENT_DIR, "datasets/PartTUT/test.txt")
    elif args.task == "ner":
        TRAIN = os.path.join(CURRENT_DIR, "datasets/MITRestaurant/train.txt")
        DEV = os.path.join(CURRENT_DIR, "datasets/MITRestaurant/dev.txt")
        TEST = os.path.join(CURRENT_DIR, "datasets/MITRestaurant/test.txt")
    
    # Usar argumentos proporcionados o valores predeterminados
    train_path = args.train if args.train else TRAIN
    dev_path = args.dev if args.dev else DEV
    test_path = args.test if args.test else TEST
    
    # Ejecutar la tarea seleccionada
    try:
        if args.task == "ner":
            ner.main(args.embedding, train_path, dev_path, test_path, args.bidirectional)
        elif args.task == "pos":
            pos.main(args.embedding, train_path, dev_path, test_path, args.bidirectional)
    except Exception as e:
        print(f"Error: {e}")
        print("Por favor, verifica las rutas a los conjuntos de datos y la tarea especificada.")