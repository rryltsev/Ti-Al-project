import numpy as np
from ase import Atoms
from lammps import lammps
from typing import Any
from ase.io import write

class ExternalASEForce:
    """
    Оптимизированный класс для связи LAMMPS fix external с ASE калькуляторами.
    Передает только координаты и ячейку, корректно масштабирует статический вириал.
    """
    
    def __init__(self, atoms: Atoms, calculator: Any, fix_id: str = "ext"):
        self.atoms = atoms
        self.calc = calculator
        self.fix_id = fix_id
        
        # Прикрепляем калькулятор к базовой структуре
        self.atoms.calc = self.calc
        self.count = 0
         
    def update_cell(self, lmp):
        """Обновление триклинной ячейки из LAMMPS в ASE"""
        box = lmp.extract_box()
        lo = np.array(box[0])
        hi = np.array(box[1])
        xy, yz, xz = box[2], box[3], box[4]
        pbc = box[5]

        # Строим матрицу ячейки по стандартам LAMMPS/ASE
        cell = np.zeros((3, 3))
        cell[0, 0] = hi[0] - lo[0]
        cell[1, 0] = xy
        cell[1, 1] = hi[1] - lo[1]
        cell[2, 0] = xz
        cell[2, 1] = yz
        cell[2, 2] = hi[2] - lo[2]

        self.atoms.set_cell(cell, scale_atoms=False)
        self.atoms.set_pbc(pbc)

    def callback(self, caller, ntimestep, nlocal, tag, x, f):
        lmp = caller
        self.count += 1
        indexes = tag - 1
        # 1. Синхронизируем размеры ячейки
        self.update_cell(lmp)

        self.atoms.positions[indexes] = x
        #self.atoms.set_positions(x)
        
        # 4. Запускаем расчет MLIP (скоростей в atoms нет -> кинетика не считается)
        energy = self.atoms.get_potential_energy()
        ase_forces = self.atoms.get_forces()
        ase_stress = self.atoms.get_stress(voigt=True) 
        
        # 5. Возвращаем силы локальным атомам текущего процессора
        f[:] = ase_forces[indexes]
        
        # 6. Передаем глобальную потенциальную энергию
        lmp.fix_external_set_energy_global(self.fix_id, energy)
        
        # 7. Конвертируем статический стресс в глобальный вириал LAMMPS
        volume = self.atoms.get_volume()
        
        virial_lammps = -ase_stress * volume 

        # Переставляем Voigt компоненты:
        # ASE:    0=xx, 1=yy, 2=zz, 3=yz, 4=xz, 5=xy
        # LAMMPS: 0=xx, 1=yy, 2=zz, 3=xy, 4=xz, 5=yz
        lammps_virial = np.array([
            virial_lammps[0], virial_lammps[1], virial_lammps[2], # xx, yy, zz
            virial_lammps[5], virial_lammps[4], virial_lammps[3]  # xy, xz, yz
        ])
        
        lmp.fix_external_set_virial_global(self.fix_id, lammps_virial)

    def register(self, lmp: lammps):
        """Регистрация колбека в инстансе LAMMPS"""
        lmp.set_fix_external_callback(self.fix_id, self.callback, lmp)
