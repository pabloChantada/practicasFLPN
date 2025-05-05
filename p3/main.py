'''
Mirar como implementamos, si con main() o con una clase
'''


import argparse
import named_entity as ner 
import pos_tagging as pos

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for NER or PoS tagging.")
    parser.add_argument("--embedding", type=str, required=True, choices=["random", "word2vec"],
                        help="Type of embedding initialization (random or word2vec).")
    parser.add_argument("--task", type=str, required=True, choices=["ner", "pos"],
                        help="Task to train the model for (ner or pos).")
    parser.add_argument("--train", type=str, required=True,
                        help="Path to the training set.")
    parser.add_argument("--dev", type=str, required=True,
                        help="Path to the development set.")
    parser.add_argument("--test", type=str, required=True,
                        help="Path to the test set.")
    args = parser.parse_args()
    
    try:
        if args.task == "ner":
            ner.main(args.embedding, args.train, args.dev, args.test)
        elif args.task == "pos":
            pos.main(args.embedding, args.train, args.dev, args.test)
    except Exception as e:
        print(f"Error: {e}")
        print("Please check the paths to the datasets and the task specified.")
