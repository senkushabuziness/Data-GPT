import pandas as pd
import re
from typing import Tuple

def get_schema_from_df(df: pd.DataFrame, table_name: str) -> Tuple[str, str, pd.Series, str, list]:
    """Generate schema and sample data from DataFrame."""
    dtype_mapping = {
        'int64': 'INTEGER',
        'int32': 'INTEGER',
        'Int64': 'INTEGER',
        'float64': 'DECIMAL(15,2)',
        'float32': 'DECIMAL(15,2)',
        'object': 'VARCHAR',
        'bool': 'BOOLEAN',
        'boolean': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'category': 'VARCHAR'
    }
    sanitized_columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col.lower()).strip('_') for col in df.columns]
    schema = ", ".join([
        f'"{col}" {dtype_mapping.get(df[orig_col].dtype.name, "VARCHAR")}'
        for col, orig_col in zip(sanitized_columns, df.columns)
    ])
    schema_sql = f'CREATE TABLE "{table_name}" ({schema});'
    sample_data = df.head(5).to_csv(index=False)
    return schema_sql, sample_data, df.dtypes, sanitized_columns[0], df[sanitized_columns[0]].dropna().astype(str).unique()[:10].tolist()

def clean_sql_output(sql_text: str) -> str:
    """Clean SQL output from LLM response."""
    sql_text = sql_text.strip()
    sql_text = re.sub(r'```(?:sql)?\s*', '', sql_text)
    sql_text = re.sub(r'```$', '', sql_text)
    sql_text = sql_text.replace('`', '"')
    sql_text = "\n".join([line for line in sql_text.splitlines() if not line.strip().startswith(("#", "--"))])
    if not sql_text.endswith(';'):
        sql_text += ';'
    match = re.search(r"(SELECT[\s\S]+?);?$", sql_text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else sql_text