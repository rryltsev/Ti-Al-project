import os
import random
import shutil
from pymatgen.core.composition import Composition
from ase.build import make_supercell
from ase.io import read, write

# Заданный список элементов системы
system_elements = ["Ti", "Al"]

# Определяем порядок элементов через Pymatgen
comp = Composition("".join(system_elements))  # Например, "TiAl"
element_order = [str(el) for el in comp.elements]  # Порядок, заданный Pymatgen
print(f"Порядок элементов, определенный Pymatgen: {element_order}")

# Параметры для каждой структуры
count_params = {
    "POSCAR_beta-Ti-bcc": {"range": 0.6, "step": 0.03},
    "POSCAR_alpha-Ti-hcp_orthogonal": {"range": 0.4, "step": 0.03},
    "POSCAR_AlTi3_D0_19": {"range": 0.15, "step": 0.03},
    "POSCAR_gamma_AlTi_L1_0": {"range": 0.15, "step": 0.03},
}

# Функция для создания суперячейки с заданным диапазоном атомов
def create_supercell(atoms, min_atoms=150, max_atoms=200):
    """
    Создает суперячейку с числом атомов в диапазоне min_atoms–max_atoms.
    """
    n = 1
    while True:
        supercell = make_supercell(atoms, [[n, 0, 0], [0, n, 0], [0, 0, n]])
        num_atoms = len(supercell)
        if num_atoms >= min_atoms and num_atoms <= max_atoms:
            return supercell
        elif num_atoms > max_atoms:
            return make_supercell(atoms, [[n, 0, 0], [0, n, 0], [0, 0, n]])
        n += 1

# Основной цикл по файлам
current_path = os.getcwd()
result_dir = os.path.join(current_path, "RESULT")  # Создаем папку RESULT
os.makedirs(result_dir, exist_ok=True)  # Создаем, если ее нет

# Проверка наличия скриптов в текущей директории
md_script = os.path.join(current_path, "mace_md_npt.py")
slurm_script = os.path.join(current_path, "run_mace_md_npt.sh")
for script in [md_script, slurm_script]:
    if not os.path.exists(script):
        print(f"Ошибка: файл {script} не найден в {current_path}. Прерываем выполнение.")
        exit()

for file_name, params in count_params.items():
    path_to_load = os.path.join(current_path, file_name)
    
    # 1. Чтение структуры
    try:
        atoms = read(path_to_load, format='vasp')
        print(f"Обрабатываем {file_name}...")
    except FileNotFoundError:
        print(f"Файл {file_name} не найден, пропускаем...")
        continue

    # 2. Создание суперячейки
    supercell = create_supercell(atoms)
    num_atoms = len(supercell)
    print(f"Создана суперячейка для {file_name} с {num_atoms} атомами")

    # 3. Создание основной папки для исходного POSCAR внутри RESULT
    main_dir = os.path.join(result_dir, file_name)
    os.makedirs(main_dir, exist_ok=True)

    # 4. Определяем элементы в исходной структуре
    initial_symbols = set(supercell.symbols)
    print(f"Исходные элементы в {file_name}: {initial_symbols}")

    # 5. Генерация структур с разным составом
    if len(initial_symbols) == 1:
        # Случай с одним элементом
        base_element = list(initial_symbols)[0]
        added_element = [el for el in element_order if el != base_element][0]
        print(f"Базовый элемент: {base_element}, добавляемый элемент: {added_element}")

        fractions = [i * params["step"] for i in range(int(params["range"] / params["step"]) + 1)]
        print(f"Доли добавляемого элемента ({added_element}): {fractions}")

        for frac in fractions:
            new_atoms = supercell.copy()
            num_to_replace = round(frac * num_atoms)
            indices = [i for i, sym in enumerate(new_atoms.symbols) if sym == base_element]
            if num_to_replace > len(indices):
                num_to_replace = len(indices)
            selected_indices = random.sample(indices, num_to_replace)
            new_atoms.symbols[selected_indices] = added_element

            # 6. Получение редуцированной формулы через Pymatgen
            comp_dict = {el: sum(1 for s in new_atoms.symbols if s == el) for el in element_order}
            comp = Composition(comp_dict)
            reduced_formula = comp.reduced_formula  # Например, "Ti7Al3"

            # 7. Создание подпапки с формулой и запись файлов
            output_dir = os.path.join(main_dir, reduced_formula)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "POSCAR")

            # 8. Сортировка атомов по порядку из element_order
            order_dict = {el: i for i, el in enumerate(element_order)}
            sorted_indices = sorted(range(len(new_atoms)), key=lambda i: order_dict[new_atoms.symbols[i]])
            sorted_atoms = new_atoms[sorted_indices]

            # 9. Запись POSCAR
            write(output_file, sorted_atoms, format='vasp')
            print(f"Сохранена структура: {output_file}")

            # 10. Копирование скриптов MD и SLURM
            shutil.copy(md_script, output_dir)
            shutil.copy(slurm_script, output_dir)
            print(f"Скопированы скрипты: mace_md_npt.py и run_mace_md_npt.sh в {output_dir}")

    else:
        # Случай с двумя элементами
        base_element = element_order[1]  # Второй элемент в порядке — базовый
        other_element = element_order[0]  # Первый элемент
        print(f"Базовый элемент: {base_element}, другой элемент: {other_element}")

        # Начальная доля базового элемента
        base_count = sum(1 for sym in supercell.symbols if sym == base_element)
        initial_frac = base_count / num_atoms
        print(f"Начальная доля базового элемента ({base_element}): {initial_frac}")

        # Диапазон изменения доли базового элемента
        min_frac = max(0.0, initial_frac - params["range"])
        max_frac = min(1.0, initial_frac + params["range"])
        fractions = [min_frac + i * params["step"] for i in range(int((max_frac - min_frac) / params["step"]) + 1)]
        print(f"Доли базового элемента ({base_element}): {fractions}")

        for frac in fractions:
            new_atoms = supercell.copy()
            desired_base_count = round(frac * num_atoms)
            current_base_count = sum(1 for sym in new_atoms.symbols if sym == base_element)

            if desired_base_count > current_base_count:
                num_to_replace = desired_base_count - current_base_count
                indices = [i for i, sym in enumerate(new_atoms.symbols) if sym == other_element]
                if num_to_replace > len(indices):
                    num_to_replace = len(indices)
                selected_indices = random.sample(indices, num_to_replace)
                new_atoms.symbols[selected_indices] = base_element
            elif desired_base_count < current_base_count:
                num_to_replace = current_base_count - desired_base_count
                indices = [i for i, sym in enumerate(new_atoms.symbols) if sym == base_element]
                if num_to_replace > len(indices):
                    num_to_replace = len(indices)
                selected_indices = random.sample(indices, num_to_replace)
                new_atoms.symbols[selected_indices] = other_element

            # 6. Получение редуцированной формулы через Pymatgen
            comp_dict = {el: sum(1 for s in new_atoms.symbols if s == el) for el in element_order}
            comp = Composition(comp_dict)
            reduced_formula = comp.reduced_formula  # Например, "TiAl"

            # 7. Создание подпапки с формулой и запись файлов
            output_dir = os.path.join(main_dir, reduced_formula)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "POSCAR")

            # 8. Сортировка атомов по порядку из element_order
            order_dict = {el: i for i, el in enumerate(element_order)}
            sorted_indices = sorted(range(len(new_atoms)), key=lambda i: order_dict[new_atoms.symbols[i]])
            sorted_atoms = new_atoms[sorted_indices]

            # 9. Запись POSCAR
            write(output_file, sorted_atoms, format='vasp')
            print(f"Сохранена структура: {output_file}")

            # 10. Копирование скриптов MD и SLURM
            shutil.copy(md_script, output_dir)
            shutil.copy(slurm_script, output_dir)
            print(f"Скопированы скрипты: mace_md_npt.py и run_mace_md_npt.sh в {output_dir}")