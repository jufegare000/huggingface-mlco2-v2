import sys
from awsglue.utils import getResolvedOptions
import awswrangler as wr
import pandas as pd
from huggingface_hub import HfApi


def etl_hf_co2_to_s3():
    print("Iniciando extracción súper-optimizada de Hugging Face...")

    # 1. Capturar el Token de los parámetros de AWS Glue
    args = getResolvedOptions(sys.argv, ['hf_token'])
    mi_token = args['hf_token']

    # 2. Inicializar la API
    api = HfApi(token=mi_token)

    # Extraer y censar al vuelo con cardData
    print("Descargando lista de modelos desde Hugging Face... (esto puede tomar un minuto)")
    models = api.list_models(cardData=True)
    co2_data = []

    for model in models:
        card = getattr(model, 'cardData', None) or {}

        # Solo procesamos si reporta emisiones
        if 'co2_eq_emissions' in card:
            emissions_raw = card['co2_eq_emissions']

            # Normalización inicial de emisiones
            if isinstance(emissions_raw, dict):
                emissions_val = emissions_raw.get("emissions")
                source = emissions_raw.get("source")
                training_type = emissions_raw.get("training_type")
                geo_location = emissions_raw.get("geographical_location")
                hardware = emissions_raw.get("hardware_used")
            else:
                emissions_val = emissions_raw
                source = None
                training_type = None
                geo_location = None
                hardware = None

            # Extracción de datasets (viene como string o lista)
            datasets_raw = card.get("datasets", [])
            if isinstance(datasets_raw, str):
                datasets = datasets_raw
            elif isinstance(datasets_raw, list):
                datasets = ", ".join([str(d) for d in datasets_raw if d])
            else:
                datasets = None

            # Construir el registro rico en datos
            co2_data.append({
                "model_id": model.id,
                "author": getattr(model, "author", None),
                "created_at": getattr(model, "createdAt", None),
                "last_modified": getattr(model, "lastModified", None),
                "downloads": getattr(model, "downloads", 0),
                "likes": getattr(model, "likes", 0),
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "library_name": getattr(model, "library_name", None),
                "language": str(card.get("language")) if card.get("language") else None,
                "license": str(card.get("license")) if card.get("license") else None,
                "datasets": datasets,
                "co2_emissions_grams": emissions_val,
                "source": source,
                "training_type": training_type,
                "geographical_location": geo_location,
                "hardware_used": hardware
            })

    df = pd.DataFrame(co2_data)
    print(f"Total de modelos con CO2 encontrados y enriquecidos: {len(df)}")

    # Limpieza de tipos para AWS Athena/Parquet
    # Forzamos todo lo que no es numérico a string para evitar errores de schema en Parquet
    string_cols = ["model_id", "author", "pipeline_tag", "library_name", "language", "license", "datasets", "source",
                   "training_type", "geographical_location", "hardware_used"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else None).astype("string")

    # Fechas a formato datetime seguro
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if "last_modified" in df.columns:
        df["last_modified"] = pd.to_datetime(df["last_modified"], errors="coerce")

    # Ruta de tu bucket
    s3_path = "s3://juanfgallo-huggingface-co2-experiment/huggingface/co2_emissions/dataset.parquet"

    # Escribir en S3
    print(f"Subiendo datos enriquecidos a {s3_path}...")
    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        mode="overwrite"
    )
    print("¡ETL completado con éxito!")


if __name__ == "__main__":
    etl_hf_co2_to_s3()