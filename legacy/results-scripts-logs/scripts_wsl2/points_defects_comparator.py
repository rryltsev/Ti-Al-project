"""
Point Defect Comparator: Compare surface and vacancy energies from DFT/article
vs ML potentials (DPA4, DPA4C, GRACE).
"""

import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# НАСТРОЙКИ (МЕНЯЙ ЗДЕСЬ)
# ============================================================

ROOT_DIR = Path("/home/katkoff/deepmd_check_pipline")

# Модель для сравнения (в формате family-variant-use_type)
# family: dpa4, dpa4c, grace
# variant: nano, mini, neo, air, plus / 2L, 3L
# use_type: pretrained, fit, finetune, finetune_half
MODE = "grace-2L-pretrained"   # ← МЕНЯЙ ЗДЕСЬ!
#MODE = "dpa4-air-pretrained"

# Если нужно сравнить с другой моделью (например, base), укажи здесь
# Если не нужна, оставь пустой строкой ""
MODE_COMPARED = "dpa4-air-finetune_half"             # например, "dpa4-neo-fit"

# Системы для сравнения
SYSTEMS = ["fcc_Al", "alpha_Ti", "gamma_AlTi", "alpha2_AlTi3", "D022_Al3Ti"]

# ============================================================
# ПУТИ
# ============================================================

# Бенчмарки (DFT/литература)
BENCHMARK_BASE = ROOT_DIR / "benchmark_data"
DATA_CSV = BENCHMARK_BASE / "surface_vacancy_energies_Ti-Al.csv"

# Результаты наших расчётов
RESULTS_BASE = ROOT_DIR / "models_check" / "results" / "point_defects"

# Выходная папка для сравнения
OUTPUT_BASE = ROOT_DIR / "compare_test" / "point_defects"

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def parse_mode(mode: str) -> tuple:
    """Разбирает MODE на family, variant, use_type."""
    parts = mode.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid MODE format: {mode}. Expected family-variant-use_type")
    return parts[0], parts[1], parts[2]


def get_model_label(family: str, variant: str, use_type: str) -> str:
    """Возвращает читаемый лейбл для модели."""
    return f"{family.upper()} {variant} ({use_type})"


def get_model_path(family: str, variant: str, use_type: str) -> Path:
    """Возвращает путь к результатам для конкретной модели."""
    return RESULTS_BASE / family / variant / use_type


def load_reference_data():
    """Load DFT reference data from CSV."""
    logger.info(f"Loading reference data from {DATA_CSV}")
    
    if not DATA_CSV.exists():
        logger.error(f"Reference CSV not found: {DATA_CSV}")
        return {}
    
    df = pd.read_csv(DATA_CSV, sep=';', usecols=[0, 1, 2], dtype=str)
    df.columns = ['Structure', 'Property', 'DFT']
    df = df[df['Structure'].isin(SYSTEMS)]
    
    ref_data = {}
    for _, row in df.iterrows():
        system = row['Structure']
        prop = row['Property']
        dft_val = row['DFT'].strip() if pd.notna(row['DFT']) and row['DFT'].strip() else None
        
        if system not in ref_data:
            ref_data[system] = {}
        
        try:
            ref_data[system][prop] = float(dft_val) if dft_val else None
        except ValueError:
            ref_data[system][prop] = None
    
    logger.info(f"Loaded DFT reference data for {len(ref_data)} systems")
    return ref_data


def load_our_results(family: str, variant: str, use_type: str, system: str) -> dict:
    """Load our calculated results from JSON."""
    json_path = RESULTS_BASE / family / variant / use_type / system / "results.json"
    
    if not json_path.exists():
        logger.warning(f"Results not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {p['property']: p['value'] for p in data.get('properties', [])}
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return {}


def build_comparison_table(ref_data, main_family, main_variant, main_use_type, 
                           comp_family=None, comp_variant=None, comp_use_type=None):
    """Build comparison dataframe with DFT and model values."""
    rows = []
    
    main_label = get_model_label(main_family, main_variant, main_use_type)
    
    if comp_family and comp_variant and comp_use_type:
        comp_label = get_model_label(comp_family, comp_variant, comp_use_type)
        has_compared = True
    else:
        comp_label = None
        has_compared = False
    
    for system in SYSTEMS:
        dft_props = ref_data.get(system, {})
        
        main_results = load_our_results(main_family, main_variant, main_use_type, system)
        
        if has_compared:
            comp_results = load_our_results(comp_family, comp_variant, comp_use_type, system)
        else:
            comp_results = {}
        
        all_props = set(dft_props.keys()) | set(main_results.keys()) | set(comp_results.keys())
        
        for prop in sorted(all_props):
            dft_val = dft_props.get(prop)
            main_val = main_results.get(prop)
            comp_val = comp_results.get(prop) if has_compared else None
            
            main_err = ((main_val - dft_val) / dft_val * 100) if (main_val is not None and dft_val is not None and dft_val != 0) else None
            comp_err = ((comp_val - dft_val) / dft_val * 100) if (comp_val is not None and dft_val is not None and dft_val != 0) else None
            
            row = {
                "System": system,
                "Property": prop,
                "DFT": dft_val,
                main_label: main_val,
                f"{main_label} Error %": round(main_err, 1) if main_err is not None else None,
            }
            
            if has_compared:
                row[comp_label] = comp_val
                row[f"{comp_label} Error %"] = round(comp_err, 1) if comp_err is not None else None
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison_table(df, main_label, comp_label=None, output_path=None):
    """Create colored table as matplotlib figure."""
    if df.empty:
        logger.warning("Empty dataframe, skipping table plot")
        return
    
    # Определяем колонки для отображения
    cols = ["System", "Property", "DFT", main_label, f"{main_label} Error %"]
    if comp_label:
        cols.extend([comp_label, f"{comp_label} Error %"])
    
    df_plot = df[cols].copy()
    
    n_rows = len(df_plot)
    
    fig, ax = plt.subplots(figsize=(16, 0.4 * (n_rows + 2)))
    ax.axis('off')
    
    headers = cols
    
    cell_text = []
    cell_colors = []
    
    for _, row in df_plot.iterrows():
        text_row = []
        colors_row = ['white'] * len(headers)
        
        for j, col in enumerate(headers):
            val = row[col]
            if pd.isna(val):
                text_row.append("-")
            elif col == "DFT" or col == main_label or (comp_label and col == comp_label):
                text_row.append(f'{val:.3f}')
            elif "Error %" in col:
                if pd.notna(val):
                    text_row.append(f'{val:.1f}')
                    # Цвет ошибки
                    if abs(val) < 5:
                        colors_row[j] = '#90EE90'  # green
                    elif abs(val) < 15:
                        colors_row[j] = '#FFFF99'  # yellow
                    else:
                        colors_row[j] = '#FFB6C1'  # red
                else:
                    text_row.append("-")
            else:
                text_row.append(str(val))
        
        cell_text.append(text_row)
        cell_colors.append(colors_row)
    
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Header style
    for j in range(len(headers)):
        table[0, j].set_facecolor('#40466e')
        table[0, j].set_text_props(weight='bold', color='white')
    
    title = f"Surface & Vacancy Energies: DFT vs {main_label}"
    if comp_label:
        title += f" vs {comp_label}"
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    if output_path is None:
        output_path = OUTPUT_BASE / f"comparison_table_{main_family}_{main_variant}_{main_use_type}.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved comparison table: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("POINT DEFECT COMPARATOR")
    logger.info("=" * 60)
    
    # Разбираем MODE
    main_family, main_variant, main_use_type = parse_mode(MODE)
    main_label = get_model_label(main_family, main_variant, main_use_type)
    logger.info(f"Main model: {main_label}")
    
    # Если есть вторая модель для сравнения
    comp_family = comp_variant = comp_use_type = None
    comp_label = None
    if MODE_COMPARED:
        comp_family, comp_variant, comp_use_type = parse_mode(MODE_COMPARED)
        comp_label = get_model_label(comp_family, comp_variant, comp_use_type)
        logger.info(f"Compared model: {comp_label}")
    
    # Выходная папка
    output_dir = OUTPUT_BASE / main_family / main_variant / main_use_type
    if MODE_COMPARED:
        output_dir = output_dir / f"vs_{comp_family}_{comp_variant}_{comp_use_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference data
    ref_data = load_reference_data()
    
    if not ref_data:
        logger.error("No reference data loaded. Exiting.")
        return
    
    # Build comparison table
    logger.info("\nBuilding comparison table...")
    df = build_comparison_table(
        ref_data, 
        main_family, main_variant, main_use_type,
        comp_family, comp_variant, comp_use_type
    )
    
    # Save CSV
    csv_path = output_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Plot table
    plot_comparison_table(df, main_label, comp_label, output_dir / "comparison_table.png")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    logger.info(f"\n{df.to_string(index=False)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()