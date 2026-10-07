"""
Exportação dos resultados pra dashboard JSON e CSVs.
"""

import json
from pathlib import Path

import pandas as pd


def export_dashboard_json(
    stats: dict,
    output_path: str | Path,
) -> Path:
    """Exporta dados do dashboard como JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, default=str, indent=2)

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
