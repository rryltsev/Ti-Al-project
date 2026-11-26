import os
from time import perf_counter
import numpy as np
import pandas as pd
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp
from ase.optimize import FIRE

class MDSimulationManager:
    def __init__(self, poscar_path, model_path, device='cpu', **kwargs):
        self.atoms = self._load_structure(poscar_path)
        self.calculator = self._init_calculator(model_path, device)
        self.atoms.calc = self.calculator
        self.start_time = perf_counter()
        self._clean_previous_files()
        self.global_step = 0
        
        # Параметры по умолчанию
        self.default_params = {
            'time_step': 1.0,           # fs
            'temperature': 2000,        # K
            'externalstress': 1.0,      # bar
            'ttime': 20.0,              # fs
            'pfactor': 1e6,             # GPa*fs²
            'md_steps': 100,
            'relax_fmax': 0.01,         # eV/Å
            'relax_steps': 1000,        # Максимальное число шагов релаксации
            'stagnation_steps': 100,    # Шаги для критерия стагнации
            'md_traj_interval': 10,
            'md_log_interval': 10,
            'md_stats_interval': 1,
            'md_xyz_interval': 20,
            'relax_traj_interval': 100,
            'relax_log_interval': 100,
            'relax_stats_interval': 1,
            'relax_xyz_interval': 100,
            'restore_params': {
                'atoms': True
            },
            'final_md_step': False
        }
        
        self.params = {**self.default_params, **kwargs}
        
        # Проверка restore_params
        valid_restore_keys = {'atoms'}
        for key in self.params['restore_params']:
            if key not in valid_restore_keys:
                raise ValueError(f"Неизвестный параметр в restore_params: {key}")
        
        # Инициализация файлов
        self.traj_file = Trajectory("traj.traj", 'a', self.atoms)
        self.log_file = "md.log"
        self.stat_file = "statistics.csv"
        self.xyz_file = "trajectory.xyz"
        
        # Состояние системы
        self.state = {
            'atoms': self.atoms.copy(),
            'velocities': None
        }
        
        self.md_state = {
            'atoms': self.atoms.copy()
        }
    
    def _load_structure(self, poscar_path):
        """Загрузка структуры из POSCAR"""
        if not os.path.exists(poscar_path):
            raise FileNotFoundError(f"POSCAR не найден по пути: {poscar_path}")
        atoms = read(poscar_path)
        print(f"Initial cell size: {atoms.get_cell().lengths()} Å, Volume: {atoms.get_volume():.2f} Å³")
        return atoms
    
    def _init_calculator(self, model_path, device):
        """Инициализация калькулятора MACE"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель MACE не найдена по пути: {model_path}")
        calc = mace_mp(model=model_path, device=device, default_dtype="float32")
        print(f"MACE calculator initialized on {device}")
        return calc
    
    def _clean_previous_files(self):
        """Очистка старых файлов результатов"""
        for fname in ["traj.traj", "md.log", "statistics.csv", "trajectory.xyz"]:
            try:
                os.remove(fname)
            except OSError:
                pass
    
    def run_md(self, params=None, continuation=False, restore=False):
        """Запуск NPT-динамики"""
        p = {**self.params, **(params or {})}
        
        # Определяем параметры
        if restore and any(p['restore_params'].values()):
            atoms = self.md_state['atoms'] if p['restore_params']['atoms'] else self.state['atoms']
        else:
            atoms = self.state['atoms']
        
        # Восстановление структуры
        self.atoms = atoms.copy()
        self.atoms.calc = self.calculator
        
        # Установка скоростей, если не заданы
        if not self.atoms.get_velocities().any():
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=p['temperature'])
            Stationary(self.atoms)
        
        # Отладка
        print(f"Start Cell size: {self.atoms.get_cell().lengths()} Å, Volume: {self.atoms.get_volume():.2f} Å³, t: {p['temperature']}, tt: {p['ttime']}, pf: {p['pfactor']}")
        
        # Настройка NPT
        dyn = NPT(
            self.atoms,
            timestep=p['time_step'] * units.fs,
            temperature_K=p['temperature'],
            externalstress=p['externalstress'] * units.bar,
            ttime=p['ttime'] * units.fs,
            pfactor=p['pfactor'] * units.GPa * (units.fs**2),
            loginterval=p['md_log_interval'],
            logfile=self.log_file,
        )
        
        #dyn.attach(lambda: self.traj_file.write(self.atoms), interval=p['md_traj_interval'])
        dyn.attach(lambda: write(self.xyz_file, self.atoms, append=True), interval=p['md_xyz_interval'])
        dyn.attach(self._record_statistics, interval=p['md_stats_interval'])
        dyn.set_fraction_traceless(0)
        dyn.run(p['md_steps'])
        
        # Сохранение состояния
        self.state.update({
            'atoms': self.atoms.copy(),
            'velocities': self.atoms.get_velocities()
        })
        
        self.md_state.update({
            'atoms': self.atoms.copy()
        })
        
        print(f"End Cell size: {self.atoms.get_cell().lengths()} Å, Volume: {self.atoms.get_volume():.2f} Å³")
        return dyn
    
    def run_relaxation(self, params=None, reset_velocities=True):
        """Запуск релаксации структуры"""
        p = {**self.params, **(params or {})}
        self.atoms.calc = self.calculator
        
        if reset_velocities:
            self.state['velocities'] = None
        
        relax = CustomFIRE(
            FrechetCellFilter(self.atoms, mask=[True, True, True, False, False, False]),
            logfile=self.log_file,
        )
        #relax.attach(lambda: self.traj_file.write(self.atoms), interval=p['relax_traj_interval'])
        relax.attach(lambda: write(self.xyz_file, self.atoms, append=True), interval=p['relax_xyz_interval'])
        relax.attach(self._record_statistics, interval=p['relax_stats_interval'])
        
        relax.run(fmax=p['relax_fmax'], steps=p['relax_steps'], stagnation_steps=p['stagnation_steps'])
        
        self.state.update({
            'atoms': self.atoms.copy()
        })
        
        return relax
    
    def run_alternating_cycles(self, num_cycles, md_params=None, relax_params=None, strategy='continuous'):
        """Запуск чередующихся циклов MD и релаксации"""
        strategies = {
            'continuous': self._cycle_strategy_continuous,
            'restore_md': self._cycle_strategy_restore_md
        }
        
        if strategy not in strategies:
            raise ValueError(f"Неизвестная стратегия: {strategy}. Доступные: {list(strategies.keys())}")
        
        print(f"\n=== Запуск {num_cycles} циклов по стратегии '{strategy}' ===")
        strategies[strategy](num_cycles, md_params, relax_params, self.params['final_md_step'])
        
        self.traj_file.close()
    
    def _cycle_strategy_continuous(self, num_cycles, md_params, relax_params, final_md_step):
        """Сценарий 1: Плавное продолжение"""
        for cycle in range(num_cycles):
            print(f"\nЦикл {cycle+1}/{num_cycles} [continuous]")
            p = {**self.params, **(md_params or {})}
            print(f"Температура: {p['temperature']} K, ttime: {p['ttime']:.1f} fs, pfactor: {p['pfactor']:.1e}")
            self.run_md(md_params, continuation=True)
            self.run_relaxation(relax_params, reset_velocities=True)
        else:
            if final_md_step:
                print("\n=== Выполнение дополнительного MD шага ===")
                p = {**self.params, **(md_params or {})}
                print(f"Температура: {p['temperature']} K, ttime: {p['ttime']:.1f} fs, pfactor: {p['pfactor']:.1e}")
                self.run_md(md_params, continuation=True)
            else:
                pass
    
    def _cycle_strategy_restore_md(self, num_cycles, md_params, relax_params, final_md_step):
        """Сценарий 2: Восстановление параметров MD после релаксации"""
        for cycle in range(num_cycles):
            print(f"\nЦикл {cycle+1}/{num_cycles} [restore_md]")
            p = {**self.params, **(md_params or {})}
            print(f"Температура: {p['temperature']} K, ttime: {p['ttime']:.1f} fs, pfactor: {p['pfactor']:.1e}")
            self.run_md(md_params, continuation=(cycle == 0), restore=(cycle != 0))
            self.run_relaxation(relax_params, reset_velocities=True)
        else:
            if final_md_step:
                print("\n=== Выполнение дополнительного MD шага ===")
                p = {**self.params, **(md_params or {})}
                print(f"Температура: {p['temperature']} K, ttime: {p['ttime']:.1f} fs, pfactor: {p['pfactor']:.1e}")
                self.run_md(md_params, continuation=False, restore=True)
            else:
                pass
    
    def _record_statistics(self):
        """Запись статистики в CSV"""
        atoms = self.atoms
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = np.mean(stress[:3])
        
        data = {
            'step': self.global_step,
            'time_ps': (perf_counter() - self.start_time) * 1e-3,
            'temperature_K': atoms.get_temperature(),
            'set_temperature_K': self.params['temperature'],
            'total_energy_eV': atoms.get_total_energy(),
            'potential_energy_eV': atoms.get_potential_energy(),
            'kinetic_energy_eV': atoms.get_kinetic_energy(),
            'volume_A3': atoms.get_volume(),
            'volume_per_atom_A3': atoms.get_volume()/len(atoms),
            'density_g_cm3': (sum(atoms.get_masses()))/atoms.get_volume()*1.660539,
            'pressure_GPa': stress_ave,
            'stress_xx_GPa': stress[0],
            'stress_yy_GPa': stress[1],
            'stress_zz_GPa': stress[2],
            'stress_xy_GPa': stress[3],
            'stress_xz_GPa': stress[4],
            'stress_yz_GPa': stress[5],
            'enthalpy_eV': (atoms.get_total_energy() + atoms.get_volume()*stress_ave*0.1),
            'max_force_eV_A': np.max(np.linalg.norm(atoms.get_forces(), axis=1)),
            'ttime_fs': self.params['ttime'],
            'pfactor_GPa_fs2': self.params['pfactor']
        }
        self.global_step += 1
        df = pd.DataFrame([data])
        header = not os.path.exists(self.stat_file)
        df.to_csv(self.stat_file, mode='a', header=header, sep=';', index=False, float_format='%.6f')

class CustomFIRE(FIRE):
    """Модифицированный алгоритм FIRE с критерием стагнации"""
    def run(self, fmax=0.05, steps=100000000, stagnation_steps=100):
        self.fmax = fmax
        step = 0
        best_force = float('inf')
        stagnation_counter = 0

        while step < steps:
            f = self.atoms.get_forces()
            self.log(f)
            self.call_observers()

            if self.converged(f):
                return True

            max_force = (f**2).sum(axis=1).max()**0.5

            if max_force < best_force:
                best_force = max_force
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_steps:
                print(f"Остановка: нет улучшений за {stagnation_steps} шагов")
                return False

            self.step(f)
            step += 1

        print("Достигнуто максимальное число шагов")
        return False

if __name__ == "__main__":
    sim_params = {
        'temperature': 2000,
        'externalstress': 1.0,
        'md_steps': 500,
        'relax_fmax': 0.03,
        'relax_steps': 200,
        'stagnation_steps': 50,
        'md_traj_interval': 100,
        'md_log_interval': 10,
        'md_stats_interval': 1,
        'md_xyz_interval': 100,
        'relax_traj_interval': 20,
        'relax_log_interval': 10,
        'relax_stats_interval': 1,
        'relax_xyz_interval': 20,
        'final_md_step': True,
        'restore_params': {
            'atoms': True
        }
    }
    
    sim = MDSimulationManager(
        poscar_path="POSCAR",
        model_path=os.path.normpath(os.path.join("/misc/home1/u2626/md_project/model", 'mace-mpa-0-medium.model')),
        device="cpu",
        **sim_params
    )
    
    strategy = 'restore_md'
    
    if strategy == 'continuous':
        print("\n=== Запуск сценария: Плавное продолжение ===")
        sim.run_alternating_cycles(
            num_cycles=10,
            md_params={'temperature': 2000},
            relax_params={'fmax': 0.03},
            strategy='continuous'
        )
    else:
        print("\n=== Запуск сценария: Восстановление MD ===")
        sim.run_alternating_cycles(
            num_cycles=3,
            md_params={'temperature': 2000},
            relax_params={'fmax': 0.03},
            strategy='restore_md'
        )