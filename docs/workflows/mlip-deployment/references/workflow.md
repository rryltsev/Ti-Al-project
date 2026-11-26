# Экспорт и эквивалентность native / ASE / LAMMPS

Исторически сравнивались одинаковые checkpoint24 в памяти, ASE и настоящем native LAMMPS
на одной A100, четыре synthetic геометрии, включая triclinic/PBC. Это тест сериализации
и реализации, а не ошибка относительно DFT. Многопроцессное масштабирование этим не проверялось.

DPA4 native LAMMPS: официальный commit a33449868448baf3c73d8eacdb2d329b13361696
(22Jul2025 Update2), PLUGIN, Intel MPI 2021.17, одна rank.
GRACE: yury-lysogorskiy/lammps commit b1f4a6c8cc4e0d0fc93d2dec3a1fcc83558062d4,
ML-PACE + TensorFlow, serial host с GPU evaluation.
Сборка и CMake flags: `MLIP-research/scripts/hpc/build-grace-lammps.py.source`.

DPA4 требует явного `plugin load /path/libdeepmd_lmpplugin.so` до pair_style deepmd.
В runtime одновременно требовались реальные Torch cu13 и native DeepMD cudart12 libraries.
Не подменяйте major-version .so через symlink. Старый pip LAMMPS падал до parsing inputs;
используйте собственный проверенный executable/shared library и согласованные Python bindings.
Активатор сохранён в `hpc/deepmd-lammps-env.sh.source`. Он содержит исторический layout
site-packages Python3.12, который нужно проверить для своего prefix.

GRACE activation и LAMMPS environment: `hpc/grace-env.sh.source`,
`hpc/grace-lammps-env.sh.source`. Для NPT/вязкости убедитесь, что выбранный pair style
действительно выдаёт virial/stress; для использованного TensorFlow GRACE нужен pair_forces.
API и pair styles меняются: первичный источник — документация конкретной установленной версии.

Заранее зафиксируйте tolerances для E, F, stress, atom order и tensor components.
DeepMD native virial → ASE stress: `sigma = -sym(virial)/V`,
ASE Voigt: xx yy zz yz xz xy. LAMMPS pressure использует другой знак и порядок;
в units metal давление в bar, время в ps. Не сравнивайте массивы без явного mapping.

Сначала run0 на идентичных координатах, затем короткий stability pilot и измерение throughput
при нужном размере системы. Новая .pt2 graph schema или GRACE экспорт требуют повторного
native/ASE/LAMMPS comparison; история успешного edge_vec пути не подтверждает новую схему.
Рецепты: `hpc/fm-deployment.py.source`, `hpc/dpa4-lammps.py.source`,
`hpc/restart-child.py.source`; runtime snapshots и большие native binaries в ZIP отсутствуют.
