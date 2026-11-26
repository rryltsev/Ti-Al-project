import os
from ase import units
from ase.md.npt import NPT
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
import pandas as pd
from time import perf_counter
from mace.calculators import mace_mp

# Фиксированный путь к модели MACE
model_name = 'mace-mpa-0-medium.model'
path_to_model = '/misc/home1/u2626/md_project/model'
path_to_load_model = os.path.join(path_to_model, model_name)
calculator = mace_mp(model=path_to_load_model, device='cpu', default_dtype="float32")

# Текущая директория (где лежит POSCAR)
current_path = os.getcwd()
poscar_file = os.path.join(current_path, "POSCAR")

# Чтение структуры
try:
    atoms = read(poscar_file, format='vasp')
    print(f"Обрабатываем POSCAR в {current_path}...")
except FileNotFoundError:
    print(f"POSCAR не найден в {current_path}, завершаем...")
    exit()

# Установка калькулятора
atoms.calc = calculator

# Имена файлов для логов
traj_name = f"traj.traj"
log_name = f"log.log"
stat_name = f"stat.csv"
xyz_name = f"xyz.xyz"

# Удаление старых файлов, если они есть
for fname in [traj_name, log_name, stat_name, xyz_name]:
    try:
        os.remove(os.path.join(current_path, fname))
    except OSError:
        pass

# Полные пути к лог-файлам
traj_filename = Trajectory(os.path.join(current_path, traj_name), 'a', atoms)
log_filename = os.path.join(current_path, log_name)
stat_filename = os.path.join(current_path, stat_name)
xyz_filename = os.path.join(current_path, xyz_name)

# Входные параметры для MD
time_step = 1.0      # fs
temperature = 1000   # K
press = 1.0          # Bar
num_md_steps = 10000 # Количество шагов
num_interval = 10     # Интервал записи логов

sigma = 1.0          # Внешнее давление в bar
ttime = 20.0         # Постоянная времени в fs
pfactor = 1e6        # Параметр баростата в GPa

print(f"ttime = {ttime:.3f}")

# Установка начальных скоростей
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
Stationary(atoms)  # Убираем общий импульс

# Настройка NPT динамики
dyn = NPT(
    atoms,
    timestep=time_step * units.fs,
    temperature_K=temperature,
    externalstress=sigma * units.bar,
    ttime=ttime * units.fs,
    pfactor=pfactor * units.GPa * (units.fs**2),    
    loginterval=num_interval,
    logfile=log_filename,
    #trajectory=traj_filename,
)

# Функция для записи XYZ
def write_xyzframe():
    dyn.atoms.write(xyz_filename, append=True)

dyn.attach(write_xyzframe, interval=num_interval)

# Функция для записи статистики
def print_dyn(a=atoms):
    imd = dyn.get_number_of_steps()
    Natoms = len(atoms)
    etot = atoms.get_total_energy() / Natoms
    epot = atoms.get_potential_energy() / Natoms
    ekin = atoms.get_kinetic_energy() / Natoms
    temp_K = atoms.get_temperature()
    vol = atoms.get_volume()
    vola = vol / Natoms
    mass = atoms.get_masses()
    Tmass = sum(mass)
    dens = (Tmass / vol) * 1.660539067173847  # g/cm^3
    stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
    stress_ave = (stress[0] + stress[1] + stress[2]) / 3.0
    entalphy = (etot + vola * stress_ave * 6.242e-3) * 96.485  # кДж/моль
    elapsed_time = perf_counter() - start_time
    data = dict(
        step=[imd],
        density=[dens],
        volume=[vola],
        energy=[etot],
        kinetic_energy=[ekin],
        potential_energy=[epot],
        temperature=[temp_K],
        stress_average=[stress_ave],
        stress1=[stress[0]],
        stress2=[stress[1]],
        stress3=[stress[2]],
        stress4=[stress[3]],
        stress5=[stress[4]],
        stress6=[stress[5]],
        entalphy=[entalphy],
        elapsed_time=[elapsed_time]
    )
    df = pd.DataFrame(data=data)
    
    if imd == 1:
        df.to_csv(stat_filename, sep=';', index=False, mode='a')
        #print(f" imd dens vol/atom Etot(eV) Epot(eV) Ekin(eV) T(K) stress(mean,xx,yy,zz,yz,xz,xy)(GPa) Entalphy elapsed_time(sec)")
    else:
        df.to_csv(stat_filename, sep=';', index=False, header=False, mode='a')
    #if imd % 100 == 0:
    #    print(f"  {imd: >3}  {dens:.3f}  {vola:.3f} {etot:.3f} {epot:.3f} {ekin:.3f} {temp_K:.2f} {stress_ave:.2f} {stress[0]:.2f} {stress[1]:.2f} {stress[2]:.2f} {stress[3]:.2f} {stress[4]:.2f} {stress[5]:.2f} {entalphy:.4f} {elapsed_time:.3f}", end="\r", flush=True)

# Привязка функций к динамике
#dyn.attach(traj_filename.write, interval=1)
dyn.attach(print_dyn, interval=num_interval)
dyn.set_fraction_traceless(0)
# Запуск MD
start_time = perf_counter()
dyn.run(num_md_steps)
traj_filename.close()

print(f"MD завершена для {current_path}")