# Harmonic baseline: начальный аудит

По сохранённым массивам bands/DOS независимо воспроизведены sorted-frequency RMSE:

- PBE: 0.209046952110 THz.
- PBEsol: 0.278098242911 THz.
- r2SCAN: 0.259449468619 THz.

Шесть DOS-интегралов равны 3 modes/atom в допуске 1e-8. Для r²SCAN: DFT singlet 5.189323502 THz, FT singlet 4.220688046 THz; отношение квадратов около 0.662. У FT singlet меняет порядок относительно doublet, поэтому сравнивать только последние отсортированные частоты неправильно.

Сохранены найденные ссылки на DFT cells, force000/001/002, ±0.01 Å, 4×4×3, Phonopy YAML и обработку. Это воспроизведение спектральной арифметики, не новый расчёт force constants или проверка mode overlaps.

Гармонические силы исходных DPA4-Neo и GRACE-2L в новом цикле не рассчитаны. Gate H0 открыт; текущий mismatch не называется forgetting. Различия DFT training/phonon protocols и отсутствие доказанной cutoff/kmesh/supercell convergence сохраняются.
