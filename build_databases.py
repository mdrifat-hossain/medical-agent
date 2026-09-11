"""
build_databases.py

Converts the three raw Kaggle CSVs (placed in ./data/) into SQLite databases
with meaningful table names and correctly inferred column types.

Expected input files in ./data/:
    heart.csv     -> from johnsmith88/heart-disease-dataset
    cancer.csv    -> from rabieelkharoua/cancer-prediction-dataset
    diabetes.csv  -> from iammustafatz/diabetes-prediction-dataset

Output:
    db/heart_disease.db   (table: heart_disease_records)
    db/cancer.db          (table: cancer_records)
    db/diabetes.db        (table: diabetes_records)

Run:
    python build_databases.py
"""

import os
import sqlite3
import pandas as pd

DATA_DIR = "data"
DB_DIR = "db"

# Pandas dtype -> SQLite column type
DTYPE_MAP = {
    "int64": "INTEGER",
    "int32": "INTEGER",
    "float64": "REAL",
    "float32": "REAL",
    "object": "TEXT",
    "bool": "INTEGER",
}


def csv_to_sqlite(csv_path: str, db_path: str, table_name: str) -> None:
    df = pd.read_csv(csv_path)

    # Clean column names (no spaces, safe for SQL)
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cols_sql = []
    for col, dtype in df.dtypes.items():
        sqlite_type = DTYPE_MAP.get(str(dtype), "TEXT")
        cols_sql.append(f'"{col}" {sqlite_type}')

    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    cur.execute(f'CREATE TABLE "{table_name}" ({", ".join(cols_sql)})')
    conn.commit()

    df.to_sql(table_name, conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print(f"✅ Built {db_path} -> table '{table_name}'  ({len(df)} rows, {len(df.columns)} cols)")


def main():
    os.makedirs(DB_DIR, exist_ok=True)

    datasets = [
        ("heart.csv", "heart_disease.db", "heart_disease_records"),
        ("cancer.csv", "cancer.db", "cancer_records"),
        ("diabetes.csv", "diabetes.db", "diabetes_records"),
    ]

    any_built = False
    for csv_name, db_name, table_name in datasets:
        csv_path = os.path.join(DATA_DIR, csv_name)
        db_path = os.path.join(DB_DIR, db_name)

        if not os.path.exists(csv_path):
            print(f"⚠️  {csv_path} not found — skipping. "
                  f"Download the dataset from Kaggle and save it as {csv_path}.")
            continue

        csv_to_sqlite(csv_path, db_path, table_name)
        any_built = True

    if not any_built:
        print("\nNo real CSVs found. You can run `python generate_sample_data.py` "
              "first to create demo data and test the pipeline end-to-end.")


if __name__ == "__main__":
    main()
