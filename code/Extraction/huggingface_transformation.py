import pandas as pd
import numpy as np
import awswrangler as wr  # Solo si lo corres en AWS Glue


def build_final_analysis_dataset():
    print("Iniciando Stage 3: Generación de métricas derivadas...")

    # 1. Leer el dataset enriquecido del Stage 2
    # Si lo corres en AWS Glue usa wr.s3.read_parquet(), si es local usa pd.read_parquet()
    s3_input_path = "s3://juanfgallo-huggingface-co2-experiment/huggingface/co2_emissions_enriched/dataset_final.parquet"
    df = wr.s3.read_parquet(s3_input_path)

    # --- CREACIÓN DE LAS NUEVAS VARIABLES ---

    # co2_reported: Booleano 1 si tiene datos de CO2, 0 si no
    df['co2_reported'] = df['co2_emissions_grams'].notna().astype(int)

    # environment: En el paper original, suele ser un alias de hardware_used (el entorno de ejecución)
    df['environment'] = df['hardware_used']

    # performance_metrics y performance_score:
    # Agrupamos las métricas extraídas (accuracy, f1, rouge) en un solo score de rendimiento general
    def get_best_metric(row):
        metrics = {'accuracy': row.get('accuracy'), 'f1': row.get('f1'), 'rouge1': row.get('rouge1')}
        valid_metrics = {k: v for k, v in metrics.items() if pd.notnull(v)}
        if not valid_metrics:
            return None, None
        # Tomar la métrica con el valor más alto
        best_metric = max(valid_metrics, key=valid_metrics.get)
        return best_metric, valid_metrics[best_metric]

    # Aplicamos la función para crear ambas columnas simultáneamente
    df[['performance_metrics', 'performance_score']] = df.apply(
        lambda row: pd.Series(get_best_metric(row)), axis=1
    )

    # domain: Mapeo heurístico basado en el pipeline_tag de Hugging Face
    def map_domain(tag):
        if not tag: return "Unknown"
        if tag in ['text-classification', 'token-classification', 'question-answering', 'summarization',
                   'text-generation', 'translation']:
            return "NLP"
        elif tag in ['image-classification', 'object-detection', 'image-segmentation', 'text-to-image']:
            return "Computer Vision"
        elif tag in ['automatic-speech-recognition', 'text-to-speech', 'audio-classification']:
            return "Audio"
        elif tag in ['image-to-text', 'visual-question-answering', 'document-question-answering']:
            return "Multimodal"
        return "Other"

    df['domain'] = df['pipeline_tag'].apply(map_domain)

    # Estandarizamos el tamaño a floats válidos para matemáticas
    df['size'] = pd.to_numeric(df['model_size_mb'], errors='coerce')
    df['datasets_size'] = pd.to_numeric(df['datasets_size'], errors='coerce')
    df['co2_eq_emissions'] = pd.to_numeric(df['co2_emissions_grams'], errors='coerce')

    # --- MÉTRICAS DE EFICIENCIA (Fórmulas del paper) ---

    # size_efficency: Gramos de CO2 por Megabyte del modelo
    df['size_efficency'] = np.where(
        df['size'] > 0,
        df['co2_eq_emissions'] / df['size'],
        None
    )

    # datasets_size_efficency: Gramos de CO2 por unidad de tamaño del dataset
    df['datasets_size_efficency'] = np.where(
        df['datasets_size'] > 0,
        df['co2_eq_emissions'] / df['datasets_size'],
        None
    )

    # 2. Filtrar y ordenar EXACTAMENTE las columnas que pediste
    columnas_finales = [
        'model_id', 'datasets', 'datasets_size', 'co2_eq_emissions', 'co2_reported',
        'source', 'training_type', 'geographical_location', 'environment',
        'performance_metrics', 'performance_score', 'downloads', 'likes',
        'library_name', 'domain', 'size', 'created_at', 'size_efficency',
        'datasets_size_efficency', 'is_autotrain'
    ]

    # Asegurar que todas existen y renombrar para encajar exacto con tu lista
    df = df.rename(columns={'model_id': 'modelId', 'is_autotrain': 'auto'})
    columnas_finales[0] = 'modelId'
    columnas_finales[-1] = 'auto'

    df_final = df[[col for col in columnas_finales if col in df.columns]]

    # 3. Guardar el resultado final de análisis
    s3_output_path = "s3://juanfgallo-huggingface-co2-experiment/huggingface/analysis_ready/dataset_analysis.csv"
    print(f"Guardando dataset listo para análisis en {s3_output_path}...")

    # Lo guardamos en CSV por si quieres mandarlo a Tableau, Excel, o tu framework de Data Viz
    wr.s3.to_csv(df_final, s3_output_path, index=False)

    print("¡Stage 3 completado! Todos los parámetros listos.")


if __name__ == "__main__":
    build_final_analysis_dataset()