#!/usr/bin/env python3
"""
SFE Comparator: Compare stacking fault energies from DFT (article)
vs ML potentials (DPA4, DPA4C, GRACE).
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
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
# use_type: pretrained, fit, finetune
#MODE = "dpa4-air-pretrained"   # ← МЕНЯЙ ЗДЕСЬ!
MODE = "grace-2L-pretrained"   # ← МЕНЯЙ ЗДЕСЬ!

# Если нужно сравнить с другой моделью, укажи здесь
# Если не нужна, оставь пустой строкой ""
MODE_COMPARED = "dpa4-air-finetune_half"             # например, "dpa4-mini-fit"

# ============================================================
# ПУТИ
# ============================================================

# Бенчмарки (DFT/литература)
BENCHMARK_BASE = ROOT_DIR / "benchmark_data"

# Результаты наших расчётов
RESULTS_BASE = ROOT_DIR / "models_check" / "results" / "planar_defects"

# Выходная папка для сравнения
OUTPUT_BASE = ROOT_DIR / "compare_test" / "planar_defects"

# ============================================================
# СТИЛИ (НЕ ТРОГАЮ)
# ============================================================

STYLES = {
    "DFT (article)":    {"color": "red",    "marker": "o", "linestyle": "-", "linewidth": 2, "markersize": 4, "markerfacecolor": "red"},
    "Grace base":       {"color": "green",  "marker": "s", "linestyle": "-", "linewidth": 2, "markersize": 4, "markerfacecolor": "green"},
    "Grace frozen":     {"color": "blue",   "marker": "d", "linestyle": "-", "linewidth": 2, "markersize": 4, "markerfacecolor": "blue"},
}

# ============================================================
# КОНФИГУРАЦИЯ ГРАФИКОВ (НЕ ТРОГАЮ)
# ============================================================

CONFIG = {
    "gamma_AlTi": {
        "comparison_111.png": [
            {
                "title": "[11-2] direction (SISF) - Fig 5a",
                "dft_file": "TiAl_SFE_111-plaine_[112]_2-direction_Fig5a.xlsx",
                "csv_path": "111_11-2_direction/gamma_line.csv",
            },
            {
                "title": "Combined [1-10]/2 + v*[11-2]/2 (APB/CSF) - Fig 5b",
                "dft_file": "TiAl_SFE_111-plaine_[112]_2-direction_Fig5b.xlsx",
                "csv_path": "111_combined_APB_CSF/gamma_line.csv",
            },
            {
                "title": "[1-10] direction - Fig 5c",
                "dft_file": "TiAl_SFE_111-plaine_[110]_2-direction_Fig5c.xlsx",
                "csv_path": "111_1-10_direction/gamma_line.csv",
            },
        ],
    },
    "alpha2_Ti3Al": {
        "comparison_0001.png": [
            {
                "title": "[01-10] direction (SISF/APB/CSF) - Fig 6a",
                "dft_file": "Ti3Al_SFE_0001-plane_0110-direction_Fig6a.xlsx",
                "csv_path": "0001_01-10_direction/gamma_line.csv",
            },
            {
                "title": "Combined path (APB/CSF) - Fig 6b",
                "dft_file": "Ti3Al_SFE_0001-plane_0110_2-direction_Fig6b.xlsx",
                "csv_path": "0001_combined_path/gamma_line.csv",
            },
            {
                "title": "[-2110] direction - Fig 6c",
                "dft_file": "Ti3Al_SFE_0001-plane_2110_3-direction_Fig6c.xlsx",
                "csv_path": "0001_-2110_direction/gamma_line.csv",
            },
        ],
    },
}

# ============================================================
# ФУНКЦИИ (ТОЛЬКО ПУТИ)
# ============================================================

def load_dft(filepath):
    """Load DFT data from Excel file (2 columns, no headers)."""
    try:
        df = pd.read_excel(filepath, header=None)
        df.columns = ["shift_fraction", "gamma_mJm2"]
        logger.info(f"  Loaded DFT: {filepath} ({len(df)} points)")
        return df
    except Exception as e:
        logger.warning(f"  Failed to load DFT {filepath}: {e}")
        return None


def load_grace(csv_path):
    """Load model data from CSV."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"  Loaded model: {csv_path} ({len(df)} points)")
        return df
    except Exception as e:
        logger.warning(f"  Failed to load model {csv_path}: {e}")
        return None


def plot_comparison(system_name, output_name, subplots_config):
    """Create comparison plot for one system."""
    
    # Разбираем MODE
    main_family, main_variant, main_use_type = MODE.split('-')
    comp_family, comp_variant, comp_use_type = MODE_COMPARED.split('-')
    
    main_label = f"{main_family.upper()} {main_variant} ({main_use_type})"
    comp_label = f"{comp_family.upper()} {comp_variant} ({comp_use_type})"
    
    n_plots = len(subplots_config)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5))
    
    if n_plots == 1:
        axes = [axes]

    all_dfs = []

    for idx, cfg in enumerate(subplots_config):
        ax = axes[idx]
        
        # === DFT ===
        dft_path = BENCHMARK_BASE / cfg["dft_file"]
        df_dft = load_dft(dft_path)
        if df_dft is not None:
            ax.plot(df_dft["shift_fraction"], df_dft["gamma_mJm2"] * 1000,
                    label="DFT (article)", color=STYLES["DFT (article)"]["color"],
                    marker=STYLES["DFT (article)"]["marker"],
                    linestyle=STYLES["DFT (article)"]["linestyle"],
                    linewidth=STYLES["DFT (article)"]["linewidth"],
                    markersize=STYLES["DFT (article)"]["markersize"],
                    markerfacecolor=STYLES["DFT (article)"]["markerfacecolor"])
        
        # === Модель 1 (основная) ===
        main_path = RESULTS_BASE / main_family / main_variant / main_use_type / system_name / cfg["csv_path"]
        df_main = load_grace(main_path)
        if df_main is not None:
            ax.plot(df_main["shift_fraction"], df_main["gamma_mJm2"],
                    label=main_label, color=STYLES["Grace base"]["color"],
                    marker=STYLES["Grace base"]["marker"],
                    linestyle=STYLES["Grace base"]["linestyle"],
                    linewidth=STYLES["Grace base"]["linewidth"],
                    markersize=STYLES["Grace base"]["markersize"],
                    markerfacecolor=STYLES["Grace base"]["markerfacecolor"])
        
        # === Модель 2 (сравнения) ===
        comp_path = RESULTS_BASE / comp_family / comp_variant / comp_use_type / system_name / cfg["csv_path"]
        df_comp = load_grace(comp_path)
        if df_comp is not None:
            ax.plot(df_comp["shift_fraction"], df_comp["gamma_mJm2"],
                    label=comp_label, color=STYLES["Grace frozen"]["color"],
                    marker=STYLES["Grace frozen"]["marker"],
                    linestyle=STYLES["Grace frozen"]["linestyle"],
                    linewidth=STYLES["Grace frozen"]["linewidth"],
                    markersize=STYLES["Grace frozen"]["markersize"],
                    markerfacecolor=STYLES["Grace frozen"]["markerfacecolor"])
        
        # === СОБИРАЕМ ДАННЫЕ ДЛЯ CSV ===
        suffix = cfg["title"].split(" - ")[0].replace(" ", "_").replace("[", "").replace("]", "")
        dfs_to_save = []
        
        if df_dft is not None:
            dft_save = df_dft.copy()
            dft_save.columns = [f'shift_DFT_{suffix}', f'gamma_DFT_{suffix}']
            dfs_to_save.append(dft_save)
        if df_main is not None:
            main_save = df_main.copy()
            main_save.columns = [f'shift_{main_label}_{suffix}', f'gamma_{main_label}_{suffix}']
            dfs_to_save.append(main_save)
        if df_comp is not None:
            comp_save = df_comp.copy()
            comp_save.columns = [f'shift_{comp_label}_{suffix}', f'gamma_{comp_label}_{suffix}']
            dfs_to_save.append(comp_save)
        
        if dfs_to_save:
            all_dfs.append(pd.concat(dfs_to_save, axis=1))

        # Оформление
        ax.set_title(cfg["title"], fontsize=11, fontweight='bold')
        ax.set_xlabel("Shift fraction", fontsize=10)
        ax.set_ylabel("γ (mJ/m²)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')
    
    # Общий заголовок
    system_title = {"gamma_AlTi": "γ-TiAl {111} plane", "alpha2_Ti3Al": "α₂-Ti₃Al {0001} plane"}
    fig.suptitle(system_title.get(system_name, system_name), fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Сохраняем
    output_dir = OUTPUT_BASE / main_family / main_variant / main_use_type / f"vs_{comp_family}_{comp_variant}_{comp_use_type}" / system_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    if all_dfs:
        combined = pd.concat(all_dfs, axis=1)
        csv_path = output_dir / output_name.replace('.png', '.csv')
        combined.to_csv(csv_path, index=False, sep=';')
        logger.info(f"Saved CSV: {csv_path}")
    
    logger.info(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("SFE COMPARATOR")
    logger.info("=" * 60)
    
    for system_name, plots_config in CONFIG.items():
        logger.info(f"\nProcessing: {system_name}")
        for output_name, subplots_cfg in plots_config.items():
            logger.info(f"  Creating: {output_name}")
            plot_comparison(system_name, output_name, subplots_cfg)
    
    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()