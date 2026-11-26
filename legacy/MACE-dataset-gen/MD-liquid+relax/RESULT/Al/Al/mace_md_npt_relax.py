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
    def __init__(self, poscar_path, model_path, device='cpu'):
        self.atoms = self._load_structure(poscar_path)
        self.calculator = self._init_calculator(model_path, device)
        self.atoms.calc = self.calculator
        self.start_time = perf_counter()
        self._clean_previous_files()
        self.global_step = 0
        
        # Параметры по умолчанию
        self.default_params = {
            'time_step': 1.0,      # fs
            'temperature': 2000,   # K
            'pressure': 1.0,       # bar
            'ttime': 20.0,         # fs
            'pfactor': 1e6,        # GPa*fs²
            'md_steps': 100,
            'relax_fmax': 0.01,
            'relax_steps': 1000,
            'stagnation_steps': 100
        }
        
        # Инициализация файлов
        self.traj_file = Trajectory("traj.traj", 'a', self.atoms)
        self.log_file = "md.log"
        self.stat_file = "statistics.csv"
        self.xyz_file = "trajectory.xyz"
        
        # Состояние системы между циклами
        self.current_state = {
            'velocities': None,
            'ttime': self.default_params['ttime'],
            'pfactor': self.default_params['pfactor'],
            'last_stress': np.zeros(6),
            'last_pressure': 0.0
        }
        
    def _load_structure(self, poscar_path):
        """Загрузка структуры из POSCAR"""
        if not os.path.exists(poscar_path):
            raise FileNotFoundError(f"POSCAR не найден по пути: {poscar_path}")
        return read(poscar_path)
    
    def _init_calculator(self, model_path, device):
        """Инициализация калькулятора MACE"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель MACE не найдена по пути: {model_path}")
        return mace_mp(model=model_path, device=device, default_dtype="float32")
    
    def _clean_previous_files(self):
        """Очистка старых файлов результатов"""
        for fname in ["traj.traj", "md.log", "statistics.csv", "trajectory.xyz"]:
            try:
                os.remove(fname)
            except OSError:
                pass
    
    def _get_params(self, params):
        """Объединение пользовательских параметров с параметрами по умолчанию"""
        if params is None:
            return self.default_params.copy()
        return {**self.default_params, **params}
    
    def run_md(self, params=None, continuation=False):
        """Запуск NPT-динамики
        
        Args:
            params (dict): Параметры симуляции (переопределяют значения по умолчанию)
            continuation (bool): Продолжение с предыдущими скоростями
        """
        p = self._get_params(params)
        
        # Инициализация скоростей
        if continuation and self.current_state['velocities'] is not None:
            self.atoms.set_velocities(self.current_state['velocities'])
        else:
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=p['temperature'])
            Stationary(self.atoms)
        
        # Настройка NPT
        dyn = NPT(
            self.atoms,
            timestep=p['time_step'] * units.fs,
            temperature_K=p['temperature'],
            externalstress=p['pressure'] * units.bar,
            ttime=self._get_adaptive_ttime(p) * units.fs,
            pfactor=self._get_adaptive_pfactor(p) * units.GPa * (units.fs**2),
            loginterval=10,
            logfile=self.log_file,
        )
        
        # Привязка функций записи
        dyn.attach(lambda: write(self.xyz_file, self.atoms, append=True), interval=20)
        dyn.attach(self._record_statistics, interval=1)
        dyn.set_fraction_traceless(0)
        dyn.run(p['md_steps'])
        
        # Сохранение состояния после MD
        self.current_state.update({
            'velocities': dyn.atoms.get_velocities(),
            'ttime': dyn.ttime / units.fs,
            'pfactor': dyn.pfact / (units.GPa * units.fs**2),  # исправлено с pfactor → pfact
            'last_stress': dyn.atoms.get_stress(),
            'last_pressure': np.mean(dyn.atoms.get_stress()[:3]) / units.GPa
        })        
        return dyn
    
    def _get_adaptive_ttime(self, params):
        """Адаптивный ttime на основе заданной стратегии"""
        if callable(params.get('ttime_strategy')):
            return params['ttime_strategy'](self.current_state)
        return params['ttime']
    
    def _get_adaptive_pfactor(self, params):
        """Адаптивный pfactor на основе заданной стратегии"""
        if callable(params.get('pfactor_strategy')):
            return params['pfactor_strategy'](self.current_state)
        return params['pfactor']
    
    def run_relaxation(self, params=None, reset_velocities=True):
        """Запуск релаксации структуры
        
        Args:
            params (dict): Параметры релаксации
            reset_velocities (bool): Сброс скоростей после релаксации
        """
        p = self._get_params(params)
        
        if reset_velocities:
            self.current_state['velocities'] = None
        
        relax = CustomFIRE(
            FrechetCellFilter(self.atoms, mask=[True, True, True, False, False, False]),
            logfile=self.log_file,
        )
        relax.attach(lambda: write(self.xyz_file, self.atoms, append=True), interval=100)
        relax.attach(self._record_statistics, interval=1)
        
        relax.run(fmax = p.get('fmax', p['relax_fmax']), steps=p['relax_steps'], stagnation_steps=p['stagnation_steps'])
        return relax
    
    def run_alternating_cycles(self, num_cycles, md_params=None, relax_params=None, strategy='continue'):
        """Запуск чередующихся циклов MD и релаксации
        
        Args:
            num_cycles (int): Количество циклов
            md_params (dict): Параметры MD
            relax_params (dict): Параметры релаксации
            strategy (str): Стратегия работы ('continue', 'reset', 'adaptive')
        """
        strategies = {
            'continue': self._cycle_strategy_continue,
            'reset': self._cycle_strategy_reset,
            'adaptive': self._cycle_strategy_adaptive
        }
        
        if strategy not in strategies:
            raise ValueError(f"Неизвестная стратегия: {strategy}. Доступные: {list(strategies.keys())}")
        
        print(f"\n=== Запуск {num_cycles} циклов по стратегии '{strategy}' ===")
        strategies[strategy](num_cycles, md_params, relax_params)
        self.traj_file.close()
    
    def _cycle_strategy_continue(self, num_cycles, md_params, relax_params):
        """Сценарий 1: Плавное продолжение с сохранением всех параметров"""
        for cycle in range(num_cycles):
            print(f"\nЦикл {cycle+1}/{num_cycles} [continue]")
            p = self._get_params(md_params)
            print(f"Температура: {p['temperature']} K")
            print(f"ttime: {self.current_state['ttime']:.1f} fs, pfactor: {self.current_state['pfactor']:.1e}")
            
            self.run_md(md_params, continuation=True)
            self.run_relaxation(relax_params, reset_velocities=False)
            #self._save_checkpoint(cycle)
    
    def _cycle_strategy_reset(self, num_cycles, md_params, relax_params):
        """Сценарий 2: Частичный сброс параметров между циклами"""
        for cycle in range(num_cycles):
            print(f"\nЦикл {cycle+1}/{num_cycles} [reset]")
            # Сброс только термостата/баростата
            self.current_state.update({
                'ttime': self.default_params['ttime'],
                'pfactor': self.default_params['pfactor']
            })
            
            self.run_md(md_params, continuation=False)
            self.run_relaxation(relax_params, reset_velocities=True)
            #self._save_checkpoint(cycle)
    
    def _cycle_strategy_adaptive(self, num_cycles, md_params, relax_params):
        """Сценарий 3: Адаптивное изменение параметров по критериям"""
        for cycle in range(num_cycles):
            print(f"\nЦикл {cycle+1}/{num_cycles} [adaptive]")
            adaptive_params = self._get_adaptive_params(cycle, md_params)
            
            self.run_md(adaptive_params, continuation=True)
            self.run_relaxation(relax_params, reset_velocities=False)
            #self._save_checkpoint(cycle)
    
    def _get_adaptive_params(self, cycle, params):
        """Генерация адаптивных параметров на основе текущего состояния"""
        adaptive = params.copy() if params else {}
        
        # Пример адаптации: если давление > порога, корректируем параметры
        if 'pressure_threshold' in params:
            last_pressure = abs(self.current_state['last_pressure'])
            if last_pressure > params['pressure_threshold']:
                adaptive['ttime'] = min(100.0, self.current_state['ttime'] * 1.1)
                adaptive['pfactor'] = max(1e4, self.current_state['pfactor'] * 0.9)
                print(f"Адаптация: давление {last_pressure:.1f} GPa > порог {params['pressure_threshold']} GPa")
                print(f"Новые параметры: ttime={adaptive['ttime']:.1f} fs, pfactor={adaptive['pfactor']:.1e}")
        
        return adaptive
    
    def _record_statistics(self):
        """Запись полной статистики в CSV файл"""
        atoms = self.atoms
        stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
        stress_ave = np.mean(stress[:3])
        
        data = {
            'step': self.global_step,
            'time_ps': (perf_counter() - self.start_time) * 1e-3,
            'temperature_K': atoms.get_temperature(),
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
            'ttime_fs': self.current_state['ttime'],
            'pfactor_GPa_fs2': self.current_state['pfactor']
        }
        self.global_step += 1
        df = pd.DataFrame([data])
        header = not os.path.exists(self.stat_file)
        df.to_csv(self.stat_file, mode='a', header=header, index=False, float_format='%.6f')
    
    def _save_checkpoint(self, cycle):
        """Сохранение чекпоинта"""
        write(f"checkpoint_{cycle}.xyz", self.atoms)
        if cycle % 5 == 0:
            self.traj_file.write(self.atoms)

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

# Примеры использования всех трёх сценариев
if __name__ == "__main__":
    # Инициализация симуляции
    sim = MDSimulationManager(
        poscar_path="POSCAR",
        model_path=os.path.join("C:\\imet\\imet_learn\\modnet\\model", 'mace-mpa-0-medium.model'), 
        device="cuda"
    )
    
    print("\n=== Пример 1: Плавное продолжение ===")
    sim.run_alternating_cycles(
        num_cycles=10,
        md_params={'temperature': 2000, 'pressure': 1.0},
        relax_params={'fmax': 0.01},
        strategy='continue'
    )
    """
    print("\n=== Пример 2: Частичный сброс ===")
    sim.run_alternating_cycles(
        num_cycles=3,
        md_params={'temperature': lambda c: 2000 - c*100, 'pressure': 1.0},
        relax_params={'stagnation_steps': 50},
        strategy='reset'
    )
    
    print("\n=== Пример 3: Адаптивное изменение ===")
    sim.run_alternating_cycles(
        num_cycles=5,
        md_params={
            'temperature': 2000,
            'pressure_threshold': 5.0,
            'ttime_strategy': lambda s: s['ttime']*1.1 if s['last_pressure'] > 5.0 else s['ttime'],
            'pfactor_strategy': lambda s: s['pfactor']*0.9 if s['last_pressure'] > 5.0 else s['pfactor']
        },
        relax_params={'fmax': 0.005},
        strategy='adaptive'
    )
    """