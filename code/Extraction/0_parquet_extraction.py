import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET = "hfmlsoc/hub_weekly_snapshots"
API_URL = f"https://datasets-server.huggingface.co/parquet?dataset={DATASET}"

RAW_DIR = "data/raw/hf_snapshots/default_train"
SLIM_DIR = "data/processed/hf_model_index_shards"
FINAL_INDEX_FILE = "data/processed/hf_model_index.parquet"

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
DELETE_RAW_AFTER_SLIM = True

KEEP_COLUMNS = [
    "modelId",
    "id",
    "downloads",
    "likes",
    "createdAt",
    "lastModified",
    "tags",
    "pipeline_tag",
    "library_name",
]

def auto_workers():
    cpu_threads = os.cpu_count() or 4
    return min(max(cpu_threads, 4), 16)

def get_parquet_urls():
    r = requests.get(API_URL, timeout=60)
    r.raise_for_status()
    payload = r.json()

    parquet_files = payload.get("parquet_files", [])
    if not parquet_files:
        raise RuntimeError("No parquet_files found.")

    urls = []
    for item in parquet_files:
        if item.get("config") == "default" and item.get("split") == "train":
            urls.append((item["filename"], item["url"], item.get("size")))

    if not urls:
        raise RuntimeError("No parquet shards found for default/train.")

    return urls

def clean_model_shard(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["modelId"])

    df = df[cols].copy()

    if "modelId" not in df.columns and "id" in df.columns:
        df["modelId"] = df["id"]

    if "modelId" in df.columns and "id" in df.columns:
        df["modelId"] = df["modelId"].fillna(df["id"])

    if "modelId" not in df.columns:
        return pd.DataFrame(columns=["modelId"])

    df = df[df["modelId"].notna()]
    df["modelId"] = df["modelId"].astype("string")

    # Heurística mínima: repos de modelos suelen verse como org/model o user/model
    df = df[df["modelId"].str.contains("/", na=False)]

    for col in ["id", "pipeline_tag", "library_name"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    for col in ["downloads", "likes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def process_one_shard(file_info):
    filename, url, expected_size = file_info

    raw_file = os.path.join(RAW_DIR, filename)
    slim_file = os.path.join(SLIM_DIR, filename)

    if os.path.exists(slim_file):
        return f"SKIP-SLIM {filename}"

    session = requests.Session()
    session.headers.update({"User-Agent": "hf-snapshot-downloader/0.1"})

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SLIM_DIR, exist_ok=True)

    if not os.path.exists(raw_file):
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(raw_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
    else:
        if expected_size is not None and os.path.getsize(raw_file) != expected_size:
            os.remove(raw_file)
            with session.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(raw_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)

    df = pd.read_parquet(raw_file)
    slim_df = clean_model_shard(df)
    slim_df.to_parquet(slim_file, index=False)

    del df
    del slim_df

    if DELETE_RAW_AFTER_SLIM and os.path.exists(raw_file):
        os.remove(raw_file)

    return f"DONE {filename}"

def combine_slim_shards():
    files = sorted(
        os.path.join(SLIM_DIR, f)
        for f in os.listdir(SLIM_DIR)
        if f.endswith(".parquet")
    )

    if not files:
        raise RuntimeError("No slim parquet shards found.")

    dfs = [pd.read_parquet(f) for f in files]
    final_df = pd.concat(dfs, ignore_index=True)

    if "modelId" in final_df.columns:
        final_df = final_df.dropna(subset=["modelId"])
        final_df = final_df.drop_duplicates(subset=["modelId"])

    os.makedirs(os.path.dirname(FINAL_INDEX_FILE), exist_ok=True)
    final_df.to_parquet(FINAL_INDEX_FILE, index=False)

    return final_df

def main():
    workers = auto_workers()
    parquet_urls = get_parquet_urls()

    print(f"CPU threads detected: {os.cpu_count()}")
    print(f"Using workers: {workers}")
    print(f"Found {len(parquet_urls)} parquet shards.")

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_shard, item) for item in parquet_urls]

        for future in as_completed(futures):
            done += 1
            try:
                msg = future.result()
                print(f"[{done}/{len(parquet_urls)}] {msg}")
            except Exception as e:
                print(f"[{done}/{len(parquet_urls)}] ERROR: {e}")

    final_df = combine_slim_shards()

    print("\nFinal model index created:")
    print(FINAL_INDEX_FILE)
    print("Shape:", final_df.shape)
    print("Columns:", final_df.columns.tolist())

if __name__ == "__main__":
    main()