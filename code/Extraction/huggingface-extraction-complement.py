import sys
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import awswrangler as wr
from awsglue.utils import getResolvedOptions
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm

logger = logging.getLogger("hf_enrichment")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def normalize_to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_datasets(info) -> Optional[str]:
    card_data = getattr(info, "cardData", None) or {}

    datasets = safe_get(card_data, "datasets")
    if datasets:
        datasets = normalize_to_list(datasets)
        datasets = [str(x) for x in datasets if x is not None]
        datasets = sorted(set(datasets))
        return ", ".join(datasets) if datasets else None

    tags = getattr(info, "tags", None) or []
    datasets_from_tags = []
    for tag in tags:
        if isinstance(tag, str) and tag.startswith("dataset:"):
            datasets_from_tags.append(tag.replace("dataset:", ""))

    datasets_from_tags = sorted(set(datasets_from_tags))
    return ", ".join(datasets_from_tags) if datasets_from_tags else None


def extract_autotrain(info) -> bool:
    tags = getattr(info, "tags", None) or []
    tags_lower = {str(t).lower() for t in tags}
    return any(tag in tags_lower for tag in ["autotrain", "autonlp"])


def extract_model_size_mb(info) -> Optional[float]:
    total_size_bytes = 0
    siblings = getattr(info, "siblings", None) or []

    valid_suffixes = (
        ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".h5", ".onnx", ".msgpack"
    )

    for file in siblings:
        filename = getattr(file, "rfilename", None)
        size = getattr(file, "size", None)

        if filename and isinstance(filename, str) and filename.endswith(valid_suffixes):
            if isinstance(size, (int, float)) and size > 0:
                total_size_bytes += size

    if total_size_bytes <= 0:
        return None

    return round(total_size_bytes / (1024 * 1024), 2)


def extract_co2_metadata(info) -> Dict[str, Any]:
    card_data = getattr(info, "cardData", None) or {}
    emissions = safe_get(card_data, "co2_eq_emissions")

    result = {
        "training_type": None,
        "geographical_location": None,
        "hardware_used": None,
        "source_card": None,
    }

    if isinstance(emissions, dict):
        result["training_type"] = emissions.get("training_type")
        result["geographical_location"] = emissions.get("geographical_location")
        result["hardware_used"] = emissions.get("hardware_used")
        result["source_card"] = emissions.get("source")

    return result


def extract_metrics_from_model_index(model_index: Any) -> Dict[str, Any]:
    out = {
        "accuracy": None,
        "f1": None,
        "loss": None,
        "rouge1": None,
        "rougeL": None,
    }

    if not model_index:
        return out

    if isinstance(model_index, list):
        if not model_index:
            return out
        model_index = model_index[0]

    if not isinstance(model_index, dict):
        return out

    results = model_index.get("results")
    if not isinstance(results, list) or not results:
        return out

    first_result = results[0]
    if not isinstance(first_result, dict):
        return out

    metrics = first_result.get("metrics")
    if not isinstance(metrics, list):
        return out

    for metric in metrics:
        if not isinstance(metric, dict):
            continue

        metric_type = str(metric.get("type", "")).strip().lower()
        metric_value = metric.get("value")

        if metric_type == "accuracy":
            out["accuracy"] = metric_value
        elif metric_type == "f1":
            out["f1"] = metric_value
        elif metric_type == "loss":
            out["loss"] = metric_value
        elif metric_type == "rouge1":
            out["rouge1"] = metric_value
        elif metric_type == "rougel":
            out["rougeL"] = metric_value

    return out


def extract_metrics(info) -> Dict[str, Any]:
    card_data = getattr(info, "cardData", None) or {}

    model_index = safe_get(card_data, "model-index")
    if model_index is None:
        model_index = safe_get(card_data, "model_index")

    metrics = extract_metrics_from_model_index(model_index)

    if any(v is not None for v in metrics.values()):
        return metrics

    raw_metrics = safe_get(card_data, "metrics")
    if isinstance(raw_metrics, list):
        for item in raw_metrics:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                key = str(k).strip().lower()
                if key == "accuracy":
                    metrics["accuracy"] = v
                elif key == "f1":
                    metrics["f1"] = v
                elif key == "loss":
                    metrics["loss"] = v
                elif key == "rouge1":
                    metrics["rouge1"] = v
                elif key == "rougel":
                    metrics["rougeL"] = v

    return metrics


def sanitize_base_dataframe(df_base: pd.DataFrame) -> pd.DataFrame:
    df = df_base.copy()

    df = df[df["co2_emissions_grams"].notna()].copy()

    df["model_id"] = df["model_id"].astype("string")
    df = df[df["model_id"].notna()]
    df["model_id"] = df["model_id"].str.strip()

    df = df[df["model_id"] != ""]
    df = df[df["model_id"].str.contains("/", na=False)]

    return df


def enrich_one_model(api: HfApi, model_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(model_id, str):
        return None

    model_id = model_id.strip()
    if not model_id or "/" not in model_id:
        return None

    info = api.model_info(model_id, files_metadata=True)

    co2_meta = extract_co2_metadata(info)
    metrics = extract_metrics(info)

    return {
        "model_id": model_id,
        "created_at_enriched": getattr(info, "created_at", None) or getattr(info, "createdAt", None),
        "model_size_mb": extract_model_size_mb(info),
        "is_autotrain": extract_autotrain(info),
        "datasets": extract_datasets(info),
        "likes": getattr(info, "likes", None),
        "library_name": getattr(info, "library_name", None),
        "pipeline_tag_enriched": getattr(info, "pipeline_tag", None),
        "training_type": co2_meta["training_type"],
        "geographical_location": co2_meta["geographical_location"],
        "hardware_used": co2_meta["hardware_used"],
        "source_card": co2_meta["source_card"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "loss": metrics["loss"],
        "rouge1": metrics["rouge1"],
        "rougeL": metrics["rougeL"],
    }


def write_checkpoint(rows: List[Dict[str, Any]], path: str):
    if not rows:
        return

    checkpoint_df = pd.DataFrame(rows)

    try:
        wr.s3.delete_objects(path=path)
    except Exception as e:
        logger.warning(f"No se pudo borrar checkpoint anterior en {path}: {e}")

    wr.s3.to_parquet(
        df=checkpoint_df,
        path=path,
        dataset=False
    )
    logger.info(f"Checkpoint guardado en {path} con {len(checkpoint_df)} filas")


def enrich_co2_data():
    args = getResolvedOptions(sys.argv, ["hf_token"])

    api = HfApi(token=args["hf_token"])

    s3_input_path = "s3://juanfgallo-huggingface-co2-experiment/huggingface/co2_emissions/dataset.parquet"
    s3_output_path = "s3://juanfgallo-huggingface-co2-experiment/huggingface/co2_emissions_enriched/dataset_final.parquet"
    s3_checkpoint_path = "s3://juanfgallo-huggingface-co2-experiment/huggingface/checkpoints/enrichment_partial.parquet"
    s3_failed_output = "s3://juanfgallo-huggingface-co2-experiment/huggingface/checkpoints/enrichment_failed.csv"

    logger.info("Iniciando Stage 2: Enriquecimiento de datos")
    logger.info(f"Leyendo datos base desde {s3_input_path}")

    df_base = wr.s3.read_parquet(s3_input_path)

    required_cols = {"model_id", "co2_emissions_grams"}
    missing = required_cols - set(df_base.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en input: {missing}")

    logger.info(f"Filas leídas: {len(df_base)}")
    logger.info(f"Columnas input: {list(df_base.columns)}")

    df_base = sanitize_base_dataframe(df_base)

    logger.info(f"Filas luego de sanitizar: {len(df_base)}")

    model_ids = (
        df_base["model_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    logger.info(f"Model IDs únicos a procesar: {len(model_ids)}")

    if not model_ids:
        raise RuntimeError("No hay model_ids válidos para procesar tras la sanitización.")

    enriched_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    checkpoint_every = 100

    for i, model_id in enumerate(tqdm(model_ids), start=1):
        logger.info(f"[{i}/{len(model_ids)}] Procesando model_id={model_id!r}")

        if not isinstance(model_id, str):
            logger.warning(f"[{i}] model_id inválido por tipo: {type(model_id)} valor={model_id!r}")
            failed_rows.append({"model_id": str(model_id), "error": "invalid_type"})
            continue

        model_id = model_id.strip()
        if not model_id or "/" not in model_id:
            logger.warning(f"[{i}] model_id inválido por formato: {model_id!r}")
            failed_rows.append({"model_id": model_id, "error": "invalid_format"})
            continue

        # --- INICIO LÓGICA DE REINTENTO MEJORADA ---
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                row = enrich_one_model(api, model_id)
                if row is not None:
                    enriched_rows.append(row)
                    logger.info(f"[{i}] OK model_id={model_id}")
                else:
                    logger.warning(f"[{i}] Sin datos enriquecidos para model_id={model_id}")
                    failed_rows.append({"model_id": model_id, "error": "no_data"})

                break  # Éxito: Salimos del bucle de reintentos

            except Exception as e:
                # Convertimos el error a texto para una búsqueda genérica y robusta
                error_str = str(e).lower()

                # Si el error menciona "429", "rate limit" o "too many requests", es un bloqueo temporal
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if attempt < max_retries:
                        logger.warning(
                            f"[{i}] Rate limit de API alcanzado. Esperando 185 segundos... (Intento {attempt}/{max_retries})")
                        time.sleep(185)  # Dormir 3 minutos y 5 segundos
                    else:
                        logger.error(f"[{i}] Rate limit alcanzado tras {max_retries} intentos en model_id={model_id!r}")
                        failed_rows.append({"model_id": model_id, "error": "Max retries 429 Rate Limit"})
                        break  # Salir de los reintentos si llegamos al máximo
                else:
                    # Si no es un error de rate limit, es un error real (404, validación, etc). Lo atrapamos y salimos.
                    logger.exception(f"[{i}] Error al procesar model_id={model_id!r}: {e}")
                    failed_rows.append({"model_id": model_id, "error": f"Exception: {str(e)}"})
                    break  # Salir de los reintentos
        # --- FIN LÓGICA DE REINTENTO MEJORADA ---

        if i % checkpoint_every == 0:
            write_checkpoint(enriched_rows, s3_checkpoint_path)
            logger.info(
                f"Checkpoint en iteración {i}. "
                f"Éxitos={len(enriched_rows)} | Fallos={len(failed_rows)}"
            )

    if not enriched_rows:
        raise RuntimeError("No se obtuvo ningún registro enriquecido. Revisa token, conectividad o datos de entrada.")

    df_enriched = pd.DataFrame(enriched_rows)
    logger.info(f"Filas enriquecidas: {len(df_enriched)}")

    df_final = pd.merge(df_base, df_enriched, on="model_id", how="left")

    if "created_at" not in df_final.columns:
        df_final["created_at"] = df_final["created_at_enriched"]
    else:
        df_final["created_at"] = df_final["created_at"].fillna(df_final["created_at_enriched"])

    if "pipeline_tag" not in df_final.columns:
        df_final["pipeline_tag"] = df_final["pipeline_tag_enriched"]
    else:
        df_final["pipeline_tag"] = df_final["pipeline_tag"].fillna(df_final["pipeline_tag_enriched"])

    if "source" in df_final.columns:
        df_final["source"] = df_final["source"].fillna(df_final["source_card"])
    else:
        df_final["source"] = df_final["source_card"]

    numeric_cols = [
        "co2_emissions_grams",
        "downloads",
        "likes",
        "model_size_mb",
        "accuracy",
        "f1",
        "loss",
        "rouge1",
        "rougeL",
    ]
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    string_cols = [
        "model_id",
        "source",
        "pipeline_tag",
        "datasets",
        "library_name",
        "training_type",
        "geographical_location",
        "hardware_used",
    ]
    for col in string_cols:
        if col in df_final.columns:
            # CORRECCIÓN AQUÍ: Forzamos la serialización segura a string saltando nulos
            df_final[col] = df_final[col].apply(lambda x: str(x) if pd.notna(x) else None).astype("string")

    cols_to_drop = ["created_at_enriched", "pipeline_tag_enriched", "source_card"]
    df_final = df_final.drop(columns=[c for c in cols_to_drop if c in df_final.columns])

    logger.info(f"Guardando datos enriquecidos en {s3_output_path}")
    wr.s3.to_parquet(
        df=df_final,
        path=s3_output_path,
        dataset=True,
        mode="overwrite"
    )

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        wr.s3.to_csv(failed_df, s3_failed_output, index=False)
        logger.info(f"Fallos guardados en {s3_failed_output} con {len(failed_df)} filas")

    logger.info("Stage 2 completado correctamente")


if __name__ == "__main__":
    enrich_co2_data()