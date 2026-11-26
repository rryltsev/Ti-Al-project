"""
Phonon bands and DOS comparator for GRACE calculations
Usage: python compare_phonons.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor

# ============================================================
# НАСТРОЙКИ - МЕНЯТЬ ТОЛЬКО ЗДЕСЬ
# ============================================================

SYSTEM = "alpha2_AlTi3"        # alpha_Ti, fcc_Al, gamma_AlTi, alpha2_AlTi3
# Режим модели: <family>-<variant>-<use_type>
# family: dpa4, dpa4c, grace
# variant: nano, mini, neo, air, plus / 2L, 3L (по возрастанию сложности)
# use_type: pretrained, fit, finetune, finetune_half
MODE="dpa4c-air-finetune"   # ← МЕНЯЙ ЗДЕСЬ!
#MODE="grace-2L-pretrained"   # ← МЕНЯЙ ЗДЕСЬ!

CALCULATOR = "deepmd"        # grace, deepmd

DFT_METHOD="pbe"                # pbe, scan, sol, или ""

# Разбираем MODE
family, variant, use_type = MODE.split('-')

# Band path для каждой системы (согласно статье)
BAND_PATHS = {
    "alpha_Ti": "GKMGA",
    "fcc_Al": "GXKGL",
    "gamma_AlTi": "GXMGZRAZ",
    "alpha2_AlTi3": "GMKGALHA",
}
BAND_PATH = BAND_PATHS[SYSTEM]

def detect_phonon_bands_format(filepath):
    """
    Определяет формат файла phonon_bands.dat по заголовку.
    
    Returns:
        'old' - формат с kx, ky, kz, q_path (10 колонок)
        'new' - формат с q_path (7 колонок)
    """
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header = line.strip()
                break
        else:
            raise ValueError(f"No header line found in {filepath}")
    
    # Разбиваем заголовок на слова (колонки)
    columns = header.replace('#', '').split()
    
    # Проверяем наличие всех трех компонент kx, ky, kz
    if 'kx' in columns and 'ky' in columns and 'kz' in columns:
        return 'old'
    # Иначе проверяем наличие q_path
    elif 'q_path' in columns:
        return 'new'
    else:
        raise ValueError(f"Unknown format in {filepath}. Header: {header}")

# ============================================================
# ФУНКЦИЯ РАЗБИЕНИЯ НА СЕГМЕНТЫ
# ============================================================

def split_by_direction(x, y, threshold=0.5):
    if len(x) < 2:
        return [(x, y)]
    
    segments = []
    start = 0
    direction = 1 if x[1] > x[0] else -1
    
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        new_dir = 1 if dx > 0 else -1
        
        # Резкий скачок назад - новая ветвь (не соединяем)
        if dx < -threshold:
            segments.append((x[start:i], y[start:i]))
            start = i
            direction = new_dir
        # Смена направления (плавный изгиб) - берем предыдущую точку для непрерывности
        elif new_dir != direction:
            segments.append((x[start:i], y[start:i]))
            start = i-1  # <--- включаем предыдущую точку
            direction = new_dir
    
    segments.append((x[start:], y[start:]))
    return segments

def save_bands_to_csv(dft_branches, q_norm, model_bands, x_exp, y_exp, output_path):
    """Save all bands data to CSV: DFT branches, model branches, experiment."""
    data_dict = {}
    
    # Model bands (x общий = q_norm)
    data_dict['Model_q'] = q_norm
    for i in range(model_bands.shape[1]):
        data_dict[f'Model_branch_{i}'] = model_bands[:, i]
    
    # DFT branches
    for idx, (x, y) in enumerate(dft_branches):
        # Pad shorter arrays with NaN
        max_len = len(data_dict['Model_q'])
        x_padded = np.full(max_len, np.nan)
        y_padded = np.full(max_len, np.nan)
        x_padded[:len(x)] = x
        y_padded[:len(y)] = y
        
        data_dict[f'DFT_branch_{idx}_x'] = x_padded
        data_dict[f'DFT_branch_{idx}_y'] = y_padded
    
    # Experiment
    if x_exp is not None and y_exp is not None:
        max_len = len(data_dict['Model_q'])
        x_exp_padded = np.full(max_len, np.nan)
        y_exp_padded = np.full(max_len, np.nan)
        x_exp_padded[:len(x_exp)] = x_exp
        y_exp_padded[:len(y_exp)] = y_exp
        
        data_dict['Exp_x'] = x_exp_padded
        data_dict['Exp_y'] = y_exp_padded
    
    df_out = pd.DataFrame(data_dict)
    df_out.to_csv(output_path, index=False)
    print(f"Bands data saved: {output_path}")

# ============================================================
# ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ BANDPATH И СПЕЦИАЛЬНЫХ ТОЧЕК
# ============================================================

def get_bandpath_and_labels(system, band_path_str, npoints=200):
    """Загружает CIF, строит bandpath и возвращает нормированные координаты special points"""
    cif_path = f"/home/katkoff/deepmd_check_pipline/models_check/structures/{system}.cif"
    
    parser = CifParser(cif_path)
    structure = parser.parse_structures(primitive=True)[0]
    structure.remove_oxidation_states()
    atoms = AseAtomsAdaptor().get_atoms(structure)
    
    bandpath = atoms.cell.bandpath(band_path_str, npoints=npoints)
    x_coords, x_special, labels = bandpath.get_linear_kpoint_axis()
    
    # Нормируем на максимум (0-1)
    x_max = x_coords[-1]
    x_special_norm = [x / x_max for x in x_special]
    
    return x_special_norm, labels, atoms

# ============================================================
# ФУНКЦИЯ ДЛЯ ЧТЕНИЯ DFT BANDS ИЗ EXCEL
# ============================================================

def read_dft_bands_from_excel(file_path, sheet_name=0, skiprows=0):
    """Читает DFT bands из Excel, где данные идут попарно (x,y) по колонкам"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, skiprows=skiprows, dtype=str)
    df = df.replace(['--', '-', '', ' '], np.nan)
    df = df.dropna(how='all')
    
    branches = []
    n_cols = df.shape[1]
    
    for col in range(0, n_cols, 2):
        if col + 1 >= n_cols:
            break
        
        # Преобразуем в числа
        x_series = pd.to_numeric(df.iloc[:, col], errors='coerce')
        y_series = pd.to_numeric(df.iloc[:, col+1], errors='coerce')
        
        valid = x_series.notna() & y_series.notna()
        if valid.sum() == 0:
            continue
        
        x = x_series[valid].values
        y = y_series[valid].values
        
        if len(x) > 0:
            branches.append((x, y))
    
    return branches

# ============================================================
# ОСНОВНЫЕ ПУТИ
# ============================================================

RESULTS_BASE = "/home/katkoff/deepmd_check_pipline/models_check/results/phonons"
OUTPUT_BASE = "/home/katkoff/deepmd_check_pipline/compare_test/phonons"
BENCHMARK_BASE = "/home/katkoff/deepmd_check_pipline/benchmark_data"

# Путь к модельным файлам
if DFT_METHOD:
    model_bands_file = f"{RESULTS_BASE}/{family}/{variant}/{use_type}/{SYSTEM}/{DFT_METHOD}/phonon_bands.dat"
    model_dos_file = f"{RESULTS_BASE}/{family}/{variant}/{use_type}/{SYSTEM}/{DFT_METHOD}/phonon_dos.dat"
else:
    model_bands_file = f"{RESULTS_BASE}/{family}/{variant}/{use_type}/{SYSTEM}/phonon_bands.dat"
    model_dos_file = f"{RESULTS_BASE}/{family}/{variant}/{use_type}/{SYSTEM}/phonon_dos.dat"

# Выходные файлы
if DFT_METHOD:
    output_dir = f"{OUTPUT_BASE}/{family}/{variant}/{use_type}/{SYSTEM}/{DFT_METHOD}"
else:
    output_dir = f"{OUTPUT_BASE}/{family}/{variant}/{use_type}/{SYSTEM}"    
os.makedirs(output_dir, exist_ok=True)

if DFT_METHOD:
    bands_output = f"{output_dir}/phonon_bands_comparison_{SYSTEM}_{MODE}_{DFT_METHOD}.png"
    dos_output = f"{output_dir}/phonon_dos_comparison_{SYSTEM}_{MODE}_{DFT_METHOD}.png"
else:
    bands_output = f"{output_dir}/phonon_bands_comparison_{SYSTEM}_{MODE}.png"
    dos_output = f"{output_dir}/phonon_dos_comparison_{SYSTEM}_{MODE}.png"        

# ============================================================
# НАСТРОЙКИ ДЛЯ КАЖДОЙ СИСТЕМЫ
# ============================================================

print(f"\n{'='*50}")
print(f"SYSTEM: {SYSTEM}")
print(f"MODE: {MODE}")
print(f"BAND_PATH: {BAND_PATH}")
print(f"CALCULATOR: {CALCULATOR}")
print(f"{'='*50}\n")

# Получаем специальные точки для данного пути
x_special, special_labels, atoms = get_bandpath_and_labels(SYSTEM, BAND_PATH)
print(f"Special points at x = {[round(x, 4) for x in x_special]}")
print(f"Labels: {special_labels}")

# Параметры для разных систем
if SYSTEM == "alpha_Ti":
    excel_file = f"{BENCHMARK_BASE}/фононы+энтальпия.xlsx"
    sheet_name = "Лист1"
    dft_bands = read_dft_bands_from_excel(excel_file, sheet_name)
    
    # Экспериментальные точки (колонки 17-18)
    df_exp = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    df_exp = df_exp.iloc[1:].dropna(how='all')
    if df_exp.shape[1] >= 19:
        x_exp_raw = df_exp.iloc[:, 17].values
        y_exp_raw = df_exp.iloc[:, 18].values
        valid_exp = pd.notna(x_exp_raw) & pd.notna(y_exp_raw)
        x_exp = x_exp_raw[valid_exp].astype(float) * 2
        y_exp = y_exp_raw[valid_exp].astype(float)
    else:
        x_exp, y_exp = None, None
    
    dft_dos_file = f"{BENCHMARK_BASE}/alpha_Ti-DOS.dat"
    has_dos = True
    scale_x = 2
    has_experiment = True

elif SYSTEM == "fcc_Al":
    excel_file = f"{BENCHMARK_BASE}/Al_Phonons.xlsx"
    sheet_name = "Sheet1"
    dft_bands = read_dft_bands_from_excel(excel_file, sheet_name, skiprows=1)
    scale_x = 1
    x_exp, y_exp = None, None
    has_experiment = False
    has_dos = False

elif SYSTEM == "gamma_AlTi":
    excel_file = f"{BENCHMARK_BASE}/TiAl_phonons.xlsx"
    sheet_name = "Sheet1"
    dft_bands = read_dft_bands_from_excel(excel_file, sheet_name)
    scale_x = 1
    x_exp, y_exp = None, None
    has_experiment = False
    has_dos = False

elif SYSTEM == "alpha2_AlTi3":
    dft_bands = []
    scale_x = 1
    x_exp, y_exp = None, None
    has_experiment = False
    dft_dos_file = f"{BENCHMARK_BASE}/D019_DFT.dat"
    has_dos = True

else:
    raise ValueError(f"Unknown system: {SYSTEM}")

# ============================================================
# ЧТЕНИЕ МОДЕЛЬНЫХ ДАННЫХ С АВТООПРЕДЕЛЕНИЕМ ФОРМАТА
# ============================================================

print(f"Reading model bands from: {model_bands_file}")

# Определяем формат файла
fmt = detect_phonon_bands_format(model_bands_file)
print(f"  Detected format: {fmt}")

# Загружаем данные
data = np.loadtxt(model_bands_file, comments='#')

if fmt == 'old':
    # старый формат: kx ky kz q_path + 6 bands
    q_path = data[:, 3]
    bands = data[:, 4:]
    print(f"  Old format: {data.shape[1]} columns, q_path column 3, bands columns 4+")
elif fmt == 'new':
    # новый формат: q_path + 6 bands
    q_path = data[:, 0]
    bands = data[:, 1:]
    print(f"  New format: {data.shape[1]} columns, q_path column 0, bands columns 1+")
else:
    raise ValueError(f"Unknown format: {fmt}")

q_norm = q_path / q_path.max()

print(f"  Model bands: {q_norm.shape[0]} q-points, {bands.shape[1]} branches")

# ============================================================
# ПОСТРОЕНИЕ BANDS
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Модель - синие линии
for i in range(bands.shape[1]):
    ax.plot(q_norm, bands[:, i], 'b-', lw=1.5, alpha=0.7, 
            label=f"{CALCULATOR}" if i == 0 else '')

# DFT ветви - цветные пунктиры
colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 
          'gray', 'olive', 'cyan', 'magenta']

for idx, (x, y) in enumerate(dft_bands):
    x_scaled = x * scale_x
    segments = split_by_direction(x_scaled, y)
    color = colors[idx % len(colors)]
    
    for seg_x, seg_y in segments:
        if len(seg_x) > 1:
            ax.plot(seg_x, seg_y, '--', lw=1.5, alpha=0.7, color=color)
    
    ax.plot([], [], '--', lw=1.5, color=color, label=f'DFT branch {idx+1}')

# Эксперимент (только для alpha_Ti)
if has_experiment and x_exp is not None:
    ax.scatter(x_exp, y_exp, c='black', s=20, alpha=0.8, label='Experiment')
    print(f"Experimental points: {len(x_exp)}")

# Special k-точки (из ASE)
for x, label in zip(x_special, special_labels):
    if 0 <= x <= 1:
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        # Метки под графиком, в относительных координатах
        ax.text(x, -0.08, label, ha='center', va='top', fontsize=10, transform=ax.transAxes)

# Оформление
ax.set_xlabel('Wave vector q (normed)', fontsize=12)
ax.set_ylabel('Frequency (THz)', fontsize=12)

if has_experiment:
    title = f'Phonon band structure: {CALCULATOR.upper()} ({MODE} {DFT_METHOD}) vs DFT vs Experiment'
else:
    title = f'Phonon band structure: {CALCULATOR.upper()} ({MODE} {DFT_METHOD}) vs DFT'

ax.set_title(title, fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=8, ncol=2)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(bands_output, dpi=300)
plt.close()

print(f"Bands comparison saved: {bands_output}")

# Save bands data to CSV
if DFT_METHOD:
    csv_bands_path = f"{output_dir}/phonon_bands_data_{SYSTEM}_{MODE}_{DFT_METHOD}.csv"
else:    
    csv_bands_path = f"{output_dir}/phonon_bands_data_{SYSTEM}_{MODE}.csv"
save_bands_to_csv(dft_bands, q_norm, bands, x_exp, y_exp, csv_bands_path)

# ============================================================
# ПОСТРОЕНИЕ DOS (только для alpha_Ti)
# ============================================================

if has_dos:
    # Чтение модельного DOS
    dos_model = np.loadtxt(model_dos_file, comments='#')
    energy_model = dos_model[:, 0]
    dos_model_vals = dos_model[:, 1]
    
    # Чтение DFT DOS
    if os.path.exists(dft_dos_file):
        dos_dft = np.loadtxt(dft_dos_file, comments='#')
        energy_dft = dos_dft[:, 0]
        dos_dft_vals = dos_dft[:, 1]
        
        # Нормировка на максимум
        dos_model_vals = dos_model_vals / np.max(dos_model_vals)
        dos_dft_vals = dos_dft_vals / np.max(dos_dft_vals)
        
        # Построение
        fig_dos, ax_dos = plt.subplots(figsize=(8, 6))
        
        ax_dos.plot(energy_model, dos_model_vals, 'b-', lw=2, 
                    label=f'{CALCULATOR.upper()} ({MODE} {DFT_METHOD})')
        ax_dos.fill_between(energy_model, dos_model_vals, alpha=0.3, color='blue')
        
        ax_dos.plot(energy_dft, dos_dft_vals, 'r--', lw=2, label='DFT')
        ax_dos.fill_between(energy_dft, dos_dft_vals, alpha=0.3, color='red')
        
        ax_dos.set_xlabel('Frequency (THz)', fontsize=12)
        ax_dos.set_ylabel('DOS (normalized)', fontsize=12)
        ax_dos.set_title(f'Phonon DOS: {CALCULATOR.upper()} ({MODE} {DFT_METHOD}) vs DFT', fontsize=14)
        ax_dos.grid(True, alpha=0.3)
        ax_dos.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(dos_output, dpi=300)
        plt.close()
        
        print(f"DOS comparison saved: {dos_output}")
    else:
        print(f"DFT DOS file not found: {dft_dos_file}")
else:
    print(f"No DFT DOS data for {SYSTEM}, skipping DOS plot")

print(f"\n{'='*50}")
print("DONE!")
print(f"Output directory: {output_dir}")
print(f"{'='*50}\n")