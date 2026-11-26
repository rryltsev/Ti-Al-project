import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
import glob
import shutil
import logging
from dataclasses import dataclass, field
from typing import Union, Optional
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.ase.jobs import AseRelaxMaker
from atomate2.forcefields.jobs import forcefield_job
from atomate2.forcefields.schemas import ForceFieldTaskDocument
from ase.calculators.calculator import Calculator
from jobflow import run_locally, SETTINGS

from utils import CalculatorFactory, StructureFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CustomForceFieldRelaxMaker(AseRelaxMaker):
    name: str = "Custom force field relax"
    checkpoint_path: Union[str, Path] = ""
    cpu: bool = False
    default_dtype: str = "float64"
    force_field_name: str = 'MACECalculator'
    relax_cell: bool = True
    fix_symmetry: bool = False
    symprec: float | None = 1e-2
    steps: int = 500
    relax_kwargs: dict = field(default_factory=dict)
    optimizer_kwargs: dict = field(default_factory=dict)
    calculator_kwargs: dict = field(default_factory=dict)
    task_document_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        logger.info(f"Initialized CustomForceFieldRelaxMaker with calculator_type={self.force_field_name}")

    @property
    def calculator(self) -> Calculator:
        logger.info("=== CALCULATOR CREATION START ===")
        logger.info(f"force_field_name = {self.force_field_name}")
        logger.info(f"checkpoint_path = {self.checkpoint_path}")
        logger.info(f"cpu = {self.cpu}")
        logger.info(f"default_dtype = {self.default_dtype}")

        device = 'cuda' if not self.cpu else 'cpu'
        logger.info(f"device = {device}")

        factory = CalculatorFactory()
        logger.info("CalculatorFactory created")

        factory.configure(
            model_path=self.checkpoint_path,
            model_name=None,
            device=device,
            dtype=self.default_dtype
        )
        logger.info("CalculatorFactory configured")

        if self.force_field_name == "MACECalculator":
            logger.info("Creating MACE calculator")
            calc = factory.calculator("mace")
        elif self.force_field_name in ['deepmd', 'deepmd_dp4']:
            logger.info("Creating DeepMD calculator")
            calc = factory.calculator("deepmd")
        elif self.force_field_name == "grace":
            logger.info("Creating GRACE calculator")
            calc = factory.calculator("grace")
        else:
            raise ValueError(f"Unknown calculator type: {self.force_field_name}")

        logger.info(f"Calculator created: {type(calc)}")
        logger.info("=== CALCULATOR CREATION END ===")

        return calc

    @forcefield_job
    def make(
        self,
        structure: Structure,
        prev_dir: Optional[Union[str, Path]] = None
    ) -> ForceFieldTaskDocument:

        logger.info("=== MAKE START ===")
        logger.info(f"Structure sites: {len(structure)}")
        logger.info(f"prev_dir: {prev_dir}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            logger.info("Calling run_ase()")
            ase_result = self.run_ase(
                structure,
                prev_dir=prev_dir
            )
            logger.info("run_ase() returned successfully")

        logger.info("Building ForceFieldTaskDocument")

        result = ForceFieldTaskDocument.from_ase_compatible_result(
            f"Custom-{self.force_field_name}",
            ase_result,
            self.steps,
            relax_kwargs=self.relax_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            relax_cell=self.relax_cell,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec if self.fix_symmetry else None,
            ionic_step_data=self.ionic_step_data,
            store_trajectory=self.store_trajectory,
            tags=self.tags,
            **self.task_document_kwargs
        )

        logger.info("ForceFieldTaskDocument created")
        logger.info("=== MAKE END ===")

        return result


class ElasticCalculator:
    def __init__(self, args):
        self.args = args
        self.systems = [s.strip() for s in args.systems.split(",")]
        self.output_dir = Path(args.workdir) / args.family / args.variant / args.use_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        
        with open(self.output_dir / "command.txt", "w") as f:
            f.write(" ".join(sys.argv))

        self._structure_factory = StructureFactory()
        self.structure = None

    '''    
    def load_structure(self):
        logger.info(f"Loading structure from {self.args.cif}")
        
        parser = CifParser(self.args.cif)
        structure = parser.parse_structures(primitive=False)[0]
        structure.remove_oxidation_states()
        
        supercell_dims = tuple(map(int, self.args.supercell.split(',')))
        structure.make_supercell(supercell_dims)
        
        self.structure = structure
        logger.info(f"Created supercell {supercell_dims}, total sites: {len(self.structure)}")
    '''

    def load_structure(self, system):
        logger.info(f"Loading structure from {system}")
        
        structure = self._structure_factory.get_structure(system, cell_type="conventional")
        
        supercell_dims = tuple(map(int, self.args.supercell.split(',')))
        structure.make_supercell(supercell_dims)
        
        self.structure = structure
        logger.info(f"Created supercell {supercell_dims}, total sites: {len(self.structure)}")

    def run_elastic_calculation(self, system):
        logger.info(f"Starting elastic calculation... in {self.output_dir / system}")
        (self.output_dir / system).mkdir(parents=True, exist_ok=True)
        os.chdir(self.output_dir / system)
        # Удаляем старые папки job_*
        job_folders = [f for f in glob.glob("job_*") if os.path.isdir(f)]
        if job_folders:
            logger.info(f"Found {len(job_folders)} job_* folders to remove")
            for folder in job_folders:
                try:
                    shutil.rmtree(folder)
                    logger.info(f"Removed folder: {folder}")
                except Exception as e:
                    logger.info(f"Failed to remove folder {folder}: {str(e)}")
        
        elastic_flow = ElasticMaker(
            bulk_relax_maker=CustomForceFieldRelaxMaker(
                force_field_name=self.args.calculator if self.args.calculator != "mace" else "MACECalculator",
                checkpoint_path=str(Path(self.args.model_path) / self.args.model_name),
                cpu=(self.args.device == "cpu"),
                default_dtype=self.args.dtype,
                relax_kwargs={"fmax": self.args.fmax},
                relax_cell=True,
                steps=self.args.relax_steps,
                fix_symmetry=self.args.fix_symmetry,
                symprec=self.args.symprec
            ),
            elastic_relax_maker=CustomForceFieldRelaxMaker(
                force_field_name=self.args.calculator if self.args.calculator != "mace" else "MACECalculator",
                checkpoint_path=str(Path(self.args.model_path) / self.args.model_name),
                cpu=(self.args.device == "cpu"),
                default_dtype=self.args.dtype,
                relax_cell=False,
                steps=self.args.relax_steps,
                fix_symmetry=self.args.fix_symmetry,
                symprec=self.args.symprec
            )
        ).make(self.structure)
        
        log_file = self.output_dir / system / f"elastic_debug_{self.args.calculator}_{system}.log"
        logger.info(f"Writing logs to {log_file}")

        # Сохраняем текущую директорию и переходим в output_dir
        original_dir = os.getcwd()
        os.chdir(self.output_dir / system)

        try:
            # Временно перенаправляем stdout в файл
            with open(log_file, "w") as f:
                sys.stdout = f
                responses = run_locally(elastic_flow, create_folders=True)
                sys.stdout = sys.__stdout__
        finally:
            # Возвращаемся обратно
            os.chdir(original_dir)
        
        # Забираем результаты из базы данных
        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "fit_elastic_tensor"},
            properties=["output.elastic_tensor", "output.derived_properties"],
            load=True,
            sort={"completed_at": -1}
        )
        
        elastic_tensor = result["output"]["elastic_tensor"]["ieee_format"]
        derived_properties = result["output"]["derived_properties"]
        
        # Сохраняем результаты
        with open(self.output_dir / system / "elastic_tensor.npy", 'wb') as f:
            np.save(f, np.array(elastic_tensor))
        with open(self.output_dir / system / "elastic_tensor.json", 'w') as f:
            json.dump(elastic_tensor, f)
        with open(self.output_dir / system / "derived_properties.json", 'w') as f:
            json.dump(derived_properties, f)
        
        logger.info(f"Results saved for {self.args.calculator}")
        logger.info(f"Elastic tensor saved to {self.output_dir / system / 'elastic_tensor.npy'}")
        logger.info(f"Derived properties saved to {self.output_dir / system / 'derived_properties.json'}")
        
    def run(self):
        logger.info("="*50)
        logger.info("ELASTIC CALCULATION WORKFLOW")
        logger.info("="*50)
        
        for system in self.systems:
            self.load_structure(system)
            self.run_elastic_calculation(system)
        
            logger.info("="*50)
            logger.info(f"CALCULATION COMPLETED")
            logger.info(f"Results saved in: {self.output_dir / system}")
            logger.info("="*50)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--calculator", required=True, choices=["mace", "grace", "deepmd"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    
    parser.add_argument("--systems", default="all", help="Comma-separated or 'all'")
    parser.add_argument("--family", required=True, choices=["dpa4", "dpa4c", "grace"])
    parser.add_argument("--variant", required=True, choices=["nano", "mini", "neo", "air", "plus", "2L", "3L"])
    parser.add_argument("--use-type", required=True, choices=["pretrained", "fit", "finetune", "finetune_half"])
    
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--gpu-name", default="CPU", help="Human-readable GPU name")

    parser.add_argument("--workdir", default="models_check")
    
    parser.add_argument("--supercell", default="4,4,4")
    
    parser.add_argument("--fmax", type=float, default=0.00001)
    parser.add_argument("--relax-steps", type=int, default=500)
    parser.add_argument("--fix-symmetry", action="store_true", default=False)
    parser.add_argument("--symprec", type=float, default=0.01)
    
    args = parser.parse_args()
    
    elastic_calc = ElasticCalculator(args)
    elastic_calc.run()


if __name__ == "__main__":
    main()