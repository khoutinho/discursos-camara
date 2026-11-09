"""
Exportação dos resultados pra dashboard JSON e CSVs.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def to_json_safe(obj):
    """Converte tipos numpy e troca NaN/infinito por None (JSON não aceita NaN)."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is pd.NA:
        return None
    return obj


def export_dashboard_json(
    stats: dict,
    output_path: str | Path,
) -> Path:
    """Exporta dados do dashboard como JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(to_json_safe(stats), f, ensure_ascii=False, default=str, indent=2, allow_nan=False)

    print(f"Dashboard JSON exportado: {output_path}")
    return output_path


def export_processed_csv(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Exporta DataFrame processado como CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"CSV processado exportado: {output_path}")
    return output_path
