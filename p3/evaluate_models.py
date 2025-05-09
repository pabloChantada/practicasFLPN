import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

"""
Script para evaluar todas las combinaciones de modelos.
Este script ejecuta las siguientes combinaciones:
- Tarea: POS, NER
- Embeddings: random, word2vec
- Arquitectura: LSTM bidireccional, LSTM simple

Los resultados se guardan en un archivo de texto y se generan gráficas
de precisión y pérdida durante el entrenamiento para cada modelo.
"""

# Directorios de conjuntos de datos
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = {
    "pos": {
        "train": os.path.join(CURRENT_DIR, "datasets/PartTUT/train.txt"),
        "dev": os.path.join(CURRENT_DIR, "datasets/PartTUT/dev.txt"),
        "test": os.path.join(CURRENT_DIR, "datasets/PartTUT/test.txt")
    },
    "ner": {
        "train": os.path.join(CURRENT_DIR, "datasets/MITRestaurant/train.txt"),
        "dev": os.path.join(CURRENT_DIR, "datasets/MITRestaurant/dev.txt"),
        "test": os.path.join(CURRENT_DIR, "datasets/MITRestaurant/test.txt")
    }
}

def run_model(task, embedding_type, bidirectional=True):
    """Ejecuta un modelo con la configuración especificada y devuelve los resultados"""
    start_time = time.time()
    
    # Seleccionar el módulo adecuado
    if task == "pos":
        import pos_tagging as tagger_module
    else:  # task == "ner"
        import named_entity as tagger_module
    
    # Obtener las rutas de los conjuntos de datos
    train_path = DATASETS[task]["train"]
    dev_path = DATASETS[task]["dev"]
    test_path = DATASETS[task]["test"]
    
    # Ejecutar el modelo
    print(f"\nEjecutando modelo: Tarea={task}, Embeddings={embedding_type}, {'BiLSTM' if bidirectional else 'LSTM'}")
    
    try:
        # Ejecutar el modelo
        if task == "pos":
            history, loss, accuracy = tagger_module.main(embedding_type, train_path, dev_path, test_path, bidirectional)
            results_per_tag = None
        else:  # task == "ner"
            history, loss, accuracy, results_per_tag = tagger_module.main(embedding_type, train_path, dev_path, test_path, bidirectional)
        
        elapsed_time = time.time() - start_time
        
        # Resultados
        results = {
            "task": task,
            "embedding_type": embedding_type,
            "bidirectional": bidirectional,
            "history": history.history,  # Guardar todo el historial de entrenamiento
            "accuracy": float(accuracy),
            "loss": float(loss),
            "time": elapsed_time
        }
        
        # Añadir resultados específicos de NER
        if task == "ner":
            results["results_per_tag"] = results_per_tag
        
        return results
    
    except Exception as e:
        print(f"Error al ejecutar modelo: {e}")
        return None

def run_all_combinations():
    """Ejecuta todas las combinaciones de modelos y devuelve los resultados"""
    results = []
    
    # Definir todas las combinaciones
    combinations = [
        # Con LSTM Bidireccional
        {"task": "pos", "embedding_type": "random", "bidirectional": True},
        {"task": "pos", "embedding_type": "word2vec", "bidirectional": True},
        {"task": "ner", "embedding_type": "random", "bidirectional": True},
        {"task": "ner", "embedding_type": "word2vec", "bidirectional": True},
        # Con LSTM simple
        {"task": "pos", "embedding_type": "random", "bidirectional": False},
        {"task": "pos", "embedding_type": "word2vec", "bidirectional": False},
        {"task": "ner", "embedding_type": "random", "bidirectional": False},
        {"task": "ner", "embedding_type": "word2vec", "bidirectional": False}
    ]
    
    # Ejecutar cada combinación
    for combo in combinations:
        result = run_model(**combo)
        if result:
            results.append(result)
    
    return results

def save_results(results):
    """Guarda los resultados en un archivo de texto"""
    filename = "results_models.txt"
    
    with open(filename, 'w') as f:
        f.write("======================================\n")
        f.write("RESULTADOS DE EVALUACIÓN DE MODELOS\n")
        f.write("======================================\n\n")
        
        # Agrupar por tarea
        for task in ["pos", "ner"]:
            task_results = [r for r in results if r["task"] == task]
            if task_results:
                f.write(f"\n{'=' * 40}\n")
                f.write(f"TAREA: {task.upper()}\n")
                f.write(f"{'=' * 40}\n\n")
                
                # Ordenar por precisión descendente
                task_results.sort(key=lambda x: x["accuracy"], reverse=True)
                
                for i, result in enumerate(task_results):
                    f.write(f"MODELO {i+1}:\n")
                    f.write(f"  Embeddings: {result['embedding_type']}\n")
                    f.write(f"  LSTM: {'Bidireccional' if result['bidirectional'] else 'Simple'}\n")
                    f.write(f"  Precisión: {result['accuracy']:.4f}\n")
                    f.write(f"  Pérdida: {result['loss']:.4f}\n")
                    f.write(f"  Tiempo: {result['time']:.2f} segundos\n")
                    
                    # Añadir resultados de nervaluate para NER
                    if task == "ner" and "results_per_tag" in result:
                        f.write("\n  Resultados de NERvaluate por entidad:\n")
                        for entity, mode_scores in result["results_per_tag"].items():
                            f.write(f"    Entidad: {entity}\n")
                            for mode in ['strict', 'exact', 'partial', 'ent_type']:
                                if mode in mode_scores:
                                    metrics = mode_scores.get(mode, {})
                                    precision = metrics.get('precision', 0)
                                    recall = metrics.get('recall', 0)
                                    f1 = metrics.get('f1', 0)
                                    f.write(f"      [{mode.title()}] P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}\n")
                    
                    f.write("\n")
    
    print(f"Resultados guardados en {filename}")
    return filename

def plot_results(results):
    """
    Genera gráficas para cada modelo mostrando la evolución del accuracy y loss
    durante el entrenamiento, más una comparativa final de todos los modelos.
    """
    # Crear carpeta 'plots' si no existe
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Para cada modelo, crear un gráfico de su evolución durante el entrenamiento
    for result in results:
        # Crear un nombre para el modelo
        model_name = f"{result['task']}_{result['embedding_type']}{'_bi' if result['bidirectional'] else ''}"
        
        # Obtener el historial de entrenamiento
        history = result['history']
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Crear la figura
        plt.figure(figsize=(10, 5))
        
        # Graficar accuracy (eje Y izquierdo)
        plt.subplot(1, 1, 1)
        plt.plot(epochs, history['accuracy'], 'b-', label='Precisión de entrenamiento')
        plt.plot(epochs, history['val_accuracy'], 'g-', label='Precisión de validación')
        plt.plot(epochs, [result['accuracy']] * len(epochs), 'r--', label=f'Precisión final: {result["accuracy"]:.4f}')
        
        # Graficar loss (mismo eje)
        plt.plot(epochs, history['loss'], 'b:', label='Pérdida de entrenamiento')
        plt.plot(epochs, history['val_loss'], 'g:', label='Pérdida de validación')
        plt.plot(epochs, [result['loss']] * len(epochs), 'r:', label=f'Pérdida final: {result["loss"]:.4f}')
        
        plt.title(f'Modelo: {model_name}\nPrecisión y Pérdida durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Guardar gráfico
        plt.tight_layout()
        filename = f"plots/{model_name}.png"
        plt.savefig(filename)
        plt.close()
        
        print(f"Gráfica guardada como {filename}")
    
    # Crear un gráfico comparativo de todos los modelos
    plt.figure(figsize=(12, 6))
    
    # Preparar datos para el gráfico
    model_names = []
    accuracies = []
    
    # Ordenar por tarea, luego por bidirectional, luego por embedding_type
    sorted_results = sorted(results, key=lambda x: (x['task'], not x['bidirectional'], x['embedding_type']))
    
    for result in sorted_results:
        model_name = f"{result['task']}-{result['embedding_type']}"
        if result["bidirectional"]:
            model_name += "-BiLSTM"
        else:
            model_name += "-LSTM"
        
        model_names.append(model_name)
        accuracies.append(result["accuracy"])
    
    # Crear gráfico de barras para precisión
    plt.bar(range(len(model_names)), accuracies, color='skyblue')
    plt.xlabel('Modelo')
    plt.ylabel('Precisión')
    plt.title('Comparación de Precisión entre Modelos')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir valores encima de las barras
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    
    # Guardar gráfico comparativo
    comparison_filename = "plots/comparacion_modelos.png"
    plt.savefig(comparison_filename)
    plt.close()
    
    print(f"Gráfica comparativa guardada como {comparison_filename}")

def main():
    print("=== Evaluación de Modelos para Etiquetado de Secuencia ===")
    print("Ejecutando todas las combinaciones de modelos...")
    
    # Ejecutar todas las combinaciones
    results = run_all_combinations()
    
    # Guardar resultados en archivo de texto
    save_results(results)
    
    # Generar gráficas
    plot_results(results)
    
    print("\nEvaluación completada.")

if __name__ == "__main__":
    main()
