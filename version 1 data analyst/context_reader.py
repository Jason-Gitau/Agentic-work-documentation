import pandas as pd

def analyze_csv(file_path):
    """Extract schema and basic stats without LLM."""
    if hasattr(file_path, 'seek'):
        file_path.seek(0)
    df = pd.read_csv(file_path)
    null_counts = df.isnull().sum()
    
    return {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "null_counts": null_counts.to_dict(),
        "null_percentages": (null_counts / len(df) * 100).to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "total_rows": len(df)
    }

