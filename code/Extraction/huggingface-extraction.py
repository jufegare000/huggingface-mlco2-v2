import sys
from awsglue.utils import getResolvedOptions
import awswrangler as wr
import pandas as pd
from huggingface_hub import HfApi


def etl_hf_co2_to_s3():
    print("Iniciando extracción optimizada de Hugging Face...")

    # 1. Capturar el Token de los parámetros de AWS Glue de forma segura
    args = getResolvedOptions(sys.argv, ['hf_token'])
    mi_token = args['hf_token']

    # 2. Inicializar la API con el Token
    api = HfApi(token=mi_token)

    # Extraer y censar al vuelo
    models = api.list_models(cardData=True)
    co2_data = []

    for model in models:
        card = getattr(model, 'cardData', None)
        if card and 'co2_eq_emissions' in card:
            emissions_raw = card['co2_eq_emissions']

            # Normalizar si viene en formato diccionario
            if isinstance(emissions_raw, dict):
                emissions_val = emissions_raw.get("emissions")
                source = emissions_raw.get("source")
            else:
                emissions_val = emissions_raw
                source = None

            co2_data.append({
                "model_id": model.id,
                "co2_emissions_grams": emissions_val,
                "source": source,
                "pipeline_tag": getattr(model, "pipeline_tag", None),
                "downloads": getattr(model, "downloads", 0),
                "last_modified": getattr(model, "lastModified", None)
            })

    df = pd.DataFrame(co2_data)
    print(f"Total de modelos con CO2 encontrados: {len(df)}")

    # Ruta de tu bucket
    s3_path = "s3://juanfgallo-huggingface-co2-experiment/huggingface/co2_emissions/dataset.parquet"

    # Escribir en S3
    print(f"Subiendo datos a {s3_path}...")
    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        mode="overwrite"
    )
    print("¡ETL completado con éxito!")


if __name__ == "__main__":
    etl_hf_co2_to_s3()