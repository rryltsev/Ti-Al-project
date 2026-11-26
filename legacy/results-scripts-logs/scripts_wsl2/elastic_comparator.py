#!/usr/bin/env python3
"""
Elastic Comparator: Compare elastic constants from DFT/article
vs ML potentials (DPA4, DPA4C, GRACE).
"""

import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

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
#MODE = "dpa4-air-pretrained"   # ← МЕНЯЙ ЗДЕСЬ!
# Если нужно сравнить с другой моделью, укажи здесь
# Если не нужна, оставь пустой строкой ""
MODE_COMPARED = "dpa4-air-finetune_half"             # например, "dpa4-mini-fit"

# Системы для сравнения
SYSTEMS = ["fcc_Al", "alpha_Ti", "gamma_AlTi", "alpha2_AlTi3", "D022_Al3Ti"]

# ============================================================
# ПУТИ
# ============================================================

# Бенчмарки (DFT/литература)
BENCHMARK_BASE = ROOT_DIR / "benchmark_data"
DATA_CSV = BENCHMARK_BASE / "elastic_constants_Ti-Al.csv"

# Результаты наших расчётов
RESULTS_BASE = ROOT_DIR / "models_check" / "results" / "elastic"

# Выходная папка для сравнения
OUTPUT_BASE = ROOT_DIR / "compare_test" / "elastic"

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


# ============================================================
# ОСТАЛЬНЫЕ ФУНКЦИИ (БЕЗ ИЗМЕНЕНИЙ)
# ============================================================

def load_reference_data():
    """Load DFT and experimental reference data from CSV."""
    logger.info(f"Loading reference data from {DATA_CSV}")
    
    df = pd.read_csv(DATA_CSV, sep=';', usecols=[0, 1, 2, 3])
    df.columns = ['Structure', 'Property', 'Expt', 'DFT']
    df = df[df['Structure'].isin(SYSTEMS)]
    
    ref_data = {}
    for system in SYSTEMS:
        sys_df = df[df['Structure'] == system]
        if not sys_df.empty:
            ref_data[system] = sys_df.set_index('Property')[['Expt', 'DFT']].to_dict('index')
    
    logger.info(f"Loaded reference data for {len(ref_data)} systems")
    return ref_data


def load_model_tensor(family, variant, use_type, system):
    """Load elastic tensor from model calculation."""
    json_path = RESULTS_BASE / family / variant / use_type / system / "elastic_tensor.json"
    
    if not json_path.exists():
        logger.warning(f"Tensor not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            tensor_data = json.load(f)
        
        if isinstance(tensor_data, list):
            tensor = np.array(tensor_data)
        else:
            tensor = np.array(tensor_data)
        
        return tensor
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return None


def load_model_derived(family, variant, use_type, system):
    """Load derived properties from model calculation."""
    json_path = RESULTS_BASE / family / variant / use_type / system / "derived_properties.json"
    
    if not json_path.exists():
        logger.warning(f"Derived properties not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {json_path}: {e}")
        return {}


def extract_property_value(prop, tensor, derived):
    """Extract property value from tensor or derived data."""
    idx_map = {"C11": (0,0), "C12": (0,1), "C13": (0,2),
               "C33": (2,2), "C44": (3,3), "C66": (5,5)}
    
    if prop.startswith("C") and "-" not in prop:
        if prop in idx_map and tensor is not None:
            i, j = idx_map[prop]
            return tensor[i][j]
    elif prop == "C12-C66" and tensor is not None:
        return tensor[0][1] - tensor[5][5]
    elif prop == "C13-C44" and tensor is not None:
        return tensor[0][2] - tensor[3][3]
    elif prop == "Bv" and derived:
        return derived.get("k_voigt", None)
    elif prop == "Gv" and derived:
        return derived.get("g_voigt", None)
    
    return None


def build_comparison_table(ref_data, main_family, main_variant, main_use_type,
                           comp_family=None, comp_variant=None, comp_use_type=None):
    """Build comparison dataframe."""
    rows = []
    
    main_label = get_model_label(main_family, main_variant, main_use_type)
    
    if comp_family and comp_variant and comp_use_type:
        comp_label = get_model_label(comp_family, comp_variant, comp_use_type)
        has_compared = True
    else:
        comp_label = None
        has_compared = False
    
    for system in SYSTEMS:
        if system not in ref_data:
            continue
        
        tensor_main = load_model_tensor(main_family, main_variant, main_use_type, system)
        derived_main = load_model_derived(main_family, main_variant, main_use_type, system)
        
        if has_compared:
            tensor_comp = load_model_tensor(comp_family, comp_variant, comp_use_type, system)
            derived_comp = load_model_derived(comp_family, comp_variant, comp_use_type, system)
        else:
            tensor_comp = None
            derived_comp = {}
        
        for prop, ref_dict in ref_data[system].items():
            expt_val = ref_dict.get("Expt")
            dft_val = ref_dict.get("DFT")
            
            try:
                expt_val = float(expt_val) if expt_val and str(expt_val).strip() else None
            except:
                expt_val = None
            try:
                dft_val = float(dft_val) if dft_val and str(dft_val).strip() else None
            except:
                dft_val = None
            
            main_val = extract_property_value(prop, tensor_main, derived_main)
            comp_val = extract_property_value(prop, tensor_comp, derived_comp) if has_compared else None
            
            main_err_expt = ((main_val - expt_val) / expt_val * 100) if (main_val is not None and expt_val and expt_val != 0) else None
            main_err_dft = ((main_val - dft_val) / dft_val * 100) if (main_val is not None and dft_val and dft_val != 0) else None
            
            row = {
                "System": system,
                "Property": prop,
                "Expt": expt_val,
                "DFT": dft_val,
                main_label: main_val,
                f"{main_label} vs Expt %": round(main_err_expt, 1) if main_err_expt is not None else None,
                f"{main_label} vs DFT %": round(main_err_dft, 1) if main_err_dft is not None else None,
            }
            
            if has_compared:
                comp_err_expt = ((comp_val - expt_val) / expt_val * 100) if (comp_val is not None and expt_val and expt_val != 0) else None
                comp_err_dft = ((comp_val - dft_val) / dft_val * 100) if (comp_val is not None and dft_val and dft_val != 0) else None
                
                row[comp_label] = comp_val
                row[f"{comp_label} vs Expt %"] = round(comp_err_expt, 1) if comp_err_expt is not None else None
                row[f"{comp_label} vs DFT %"] = round(comp_err_dft, 1) if comp_err_dft is not None else None
            
            rows.append(row)
    
    return pd.DataFrame(rows)

def wrap_header(text, max_words_per_line=2):
    """Wrap long header text into multiple lines."""
    # Если это колонка с ошибкой, обрабатываем отдельно
    if " vs Expt %" in text:
        main_part = text.replace(" vs Expt %", "").strip()
        wrapped_main = wrap_header(main_part, max_words_per_line)
        return f"{wrapped_main}\nvs Expt %"
    elif " vs DFT %" in text:
        main_part = text.replace(" vs DFT %", "").strip()
        wrapped_main = wrap_header(main_part, max_words_per_line)
        return f"{wrapped_main}\nvs DFT %"
    
    # Разбиваем по пробелам
    words = text.split()
    
    # Если слов мало — возвращаем как есть
    if len(words) <= max_words_per_line:
        return text
    
    # Разбиваем на строки по max_words_per_line слов
    lines = []
    for i in range(0, len(words), max_words_per_line):
        lines.append(" ".join(words[i:i + max_words_per_line]))
    
    return "\n".join(lines)

def plot_comparison_table(df_comparison, output_path):
    """Create colored table as matplotlib figure."""
    if df_comparison.empty:
        return
    
    df = df_comparison.dropna(subset=["DFT"]).copy()
    n_rows = len(df)
    
    # Оригинальные заголовки
    orig_headers = list(df.columns)
    
    # Заменяем длинные названия на многострочные
    headers = [wrap_header(h, max_words_per_line=2) for h in orig_headers]
    n_cols = len(headers)
    
    fig, ax = plt.subplots(figsize=(14, 0.35 * (n_rows + 2)))
    ax.axis('off')
    
    cell_text = []
    cell_colors = []
    
    for _, row in df.iterrows():
        text_row = []
        colors_row = ['white'] * n_cols
        
        for j, col in enumerate(orig_headers):
            val = row[col]
            if pd.isna(val):
                text_row.append("-")
            elif isinstance(val, float):
                text_row.append(f'{val:.1f}')
            else:
                text_row.append(str(val))
            
            if "vs" in col and pd.notna(val):
                if abs(val) < 5:
                    colors_row[j] = '#90EE90'
                elif abs(val) < 15:
                    colors_row[j] = '#FFFF99'
                else:
                    colors_row[j] = '#FFB6C1'
        
        cell_text.append(text_row)
        cell_colors.append(colors_row)
    
    # Ширина колонок
    col_widths = [0.12] + [0.08] * (n_cols - 1)
    
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Увеличиваем высоту первой строки (заголовка)
    for j in range(n_cols):
        table[0, j].set_facecolor('#40466e')
        table[0, j].set_text_props(weight='bold', color='white')
        table[0, j].set_height(0.15)
    
    ax.set_title("Elastic Constants Comparison", fontsize=13, fontweight='bold', pad=20)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved comparison table: {output_path}")


def plot_bars_for_system(system, ref_data, main_family, main_variant, main_use_type,
                         comp_family=None, comp_variant=None, comp_use_type=None,
                         output_dir=None):
    """Create bar chart for one system."""
    if system not in ref_data:
        return
    
    props = list(ref_data[system].keys())
    n_props = len(props)
    
    if n_props == 0:
        return
    
    n_cols = min(5, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    
    if n_props == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    tensor_main = load_model_tensor(main_family, main_variant, main_use_type, system)
    derived_main = load_model_derived(main_family, main_variant, main_use_type, system)
    
    if comp_family and comp_variant and comp_use_type:
        tensor_comp = load_model_tensor(comp_family, comp_variant, comp_use_type, system)
        derived_comp = load_model_derived(comp_family, comp_variant, comp_use_type, system)
    else:
        tensor_comp = None
        derived_comp = {}
    
    for idx, prop in enumerate(props):
        ax = axes[idx]
        
        ref = ref_data[system][prop]
        expt_val = ref.get("Expt")
        dft_val = ref.get("DFT")
        
        try:
            expt_val = float(expt_val) if expt_val and str(expt_val).strip() else None
        except:
            expt_val = None
        try:
            dft_val = float(dft_val) if dft_val and str(dft_val).strip() else None
        except:
            dft_val = None
        
        main_val = extract_property_value(prop, tensor_main, derived_main)
        comp_val = extract_property_value(prop, tensor_comp, derived_comp) if tensor_comp is not None else None
        
        values = {
            "Expt": expt_val,
            "DFT": dft_val,
            get_model_label(main_family, main_variant, main_use_type): main_val,
        }
        if comp_val is not None:
            values[get_model_label(comp_family, comp_variant, comp_use_type)] = comp_val
        
        labels = [k for k, v in values.items() if v is not None]
        vals = [v for v in values.values() if v is not None]
        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'][:len(labels)]
        
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, vals, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(vals)*0.01,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_ylabel("GPa", fontsize=9)
        ax.set_title(prop, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.4)
    
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f"{system} - Elastic Properties", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = OUTPUT_BASE / main_family / main_variant / main_use_type
    
    output_path = output_dir / f"{system}_bars.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved bar chart: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("ELASTIC COMPARATOR")
    logger.info("=" * 60)
    
    main_family, main_variant, main_use_type = parse_mode(MODE)
    main_label = get_model_label(main_family, main_variant, main_use_type)
    logger.info(f"Main model: {main_label}")
    
    comp_family = comp_variant = comp_use_type = None
    comp_label = None
    if MODE_COMPARED:
        comp_family, comp_variant, comp_use_type = parse_mode(MODE_COMPARED)
        comp_label = get_model_label(comp_family, comp_variant, comp_use_type)
        logger.info(f"Compared model: {comp_label}")
    
    output_dir = OUTPUT_BASE / main_family / main_variant / main_use_type
    if MODE_COMPARED:
        output_dir = output_dir / f"vs_{comp_family}_{comp_variant}_{comp_use_type}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ref_data = load_reference_data()
    
    if not ref_data:
        logger.error("No reference data loaded. Exiting.")
        return
    
    logger.info("\nBuilding comparison table...")
    df_comparison = build_comparison_table(
        ref_data,
        main_family, main_variant, main_use_type,
        comp_family, comp_variant, comp_use_type
    )
    
    csv_path = output_dir / "comparison_table.csv"
    df_comparison.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    plot_comparison_table(df_comparison, output_dir / "comparison_table.png")
    
    for system in SYSTEMS:
        if system in ref_data:
            logger.info(f"\nPlotting bar chart for {system}")
            plot_bars_for_system(
                system, ref_data,
                main_family, main_variant, main_use_type,
                comp_family, comp_variant, comp_use_type,
                output_dir
            )
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    logger.info(f"\n{df_comparison.to_string(index=False)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()