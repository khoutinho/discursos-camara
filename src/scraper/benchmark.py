#!/usr/bin/env python3
"""
Benchmark dos 3 métodos de scraping.

Usa um tema com poucos discursos pra comparar tempo, quantidade e qualidade.

Uso:
    python -m src.scraper.benchmark
"""

import time
import json
from pathlib import Path

from . import sitaqweb, api, hybrid


BENCHMARK_KEYWORDS = ["feminicídio", "feminicídios"]

SITAQ_DATA_INICIO = "01/01/2023"
SITAQ_DATA_FIM = "31/12/2024"
API_DATA_INICIO = "2023-01-01"
API_DATA_FIM = "2024-12-31"

OUTPUT_DIR = Path("output/benchmark")


def run_benchmark():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BENCHMARK DE SCRAPING")
    print(f"Keywords: {BENCHMARK_KEYWORDS}")
    print(f"Período: {SITAQ_DATA_INICIO} – {SITAQ_DATA_FIM}")
    print("=" * 60)

    results = {}

    # ── 1. SitaQWeb ──
    print("\n--- Método 1: SitaQWeb (HTML scraping) ---")
    t0 = time.time()
    sitaq_results = sitaqweb.scrape(
        keywords=BENCHMARK_KEYWORDS,
        data_inicio=SITAQ_DATA_INICIO,
        data_fim=SITAQ_DATA_FIM,
        progress_callback=lambda p, n: print(f"  Página {p}, {n} discursos"),
    )
    t_sitaq = time.time() - t0
    results["sitaqweb"] = _summarize(sitaq_results, t_sitaq)
    print(f"  Resultado: {len(sitaq_results)} discursos em {t_sitaq:.1f}s")

    # ── 2. API Dados Abertos ──
    print("\n--- Método 2: API Dados Abertos ---")
    t0 = time.time()
    api_results = api.scrape(
        keywords=BENCHMARK_KEYWORDS,
        data_inicio=API_DATA_INICIO,
        data_fim=API_DATA_FIM,
        max_workers=5,
        progress_callback=lambda done, total, found: (
            print(f"  Deputados: {done}/{total}, discursos: {found}")
            if done % 50 == 0 else None
        ),
    )
    t_api = time.time() - t0
    results["api"] = _summarize(api_results, t_api)
    print(f"  Resultado: {len(api_results)} discursos em {t_api:.1f}s")

    # ── 3. Híbrido ──
    print("\n--- Método 3: Híbrido (SitaQWeb + API) ---")
    t0 = time.time()
    hybrid_results = hybrid.scrape(
        keywords=BENCHMARK_KEYWORDS,
        data_inicio_sitaq=SITAQ_DATA_INICIO,
        data_fim_sitaq=SITAQ_DATA_FIM,
        data_inicio_api=API_DATA_INICIO,
        data_fim_api=API_DATA_FIM,
        max_workers=5,
        progress_callback=lambda stage, detail: print(f"  [{stage}] {detail}"),
    )
    t_hybrid = time.time() - t0
    results["hybrid"] = _summarize(hybrid_results, t_hybrid)
    print(f"  Resultado: {len(hybrid_results)} discursos em {t_hybrid:.1f}s")

    # ── Comparação ──
    print("\n" + "=" * 60)
    print("COMPARAÇÃO")
    print("=" * 60)
    print(f"{'Método':<15} {'Discursos':>10} {'Tempo (s)':>10} {'Com texto':>10} {'Tam. médio':>12}")
    print("-" * 60)
    for method, summary in results.items():
        print(
            f"{method:<15} {summary['total']:>10} {summary['tempo_s']:>10.1f} "
            f"{summary['com_texto']:>10} {summary['tam_medio']:>12.0f}"
        )

    # ── Amostra de qualidade do texto ──
    print("\n--- Amostra do texto (primeiros 200 chars) ---")
    for method, data in [("sitaqweb", sitaq_results), ("api", api_results), ("hybrid", hybrid_results)]:
        if data:
            sample = data[0]["Discurso"][:200]
            has_accents = any(c in sample for c in "áàâãéèêíïóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ")
            print(f"\n  [{method}] (acentos: {'SIM' if has_accents else 'NÃO'})")
            print(f"  {sample}...")

    with open(OUTPUT_DIR / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {OUTPUT_DIR / 'benchmark_results.json'}")


def _summarize(records: list[dict], elapsed: float) -> dict:
    """Resume resultados de um método."""
    texts = [r["Discurso"] for r in records if r.get("Discurso")]
    return {
        "total": len(records),
        "com_texto": len(texts),
        "tempo_s": round(elapsed, 1),
        "tam_medio": sum(len(t) for t in texts) / max(len(texts), 1),
        "has_accents": any(
            any(c in t for c in "áàâãéèêíïóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ")
            for t in texts[:10]
        ),
    }


if __name__ == "__main__":
    run_benchmark()
