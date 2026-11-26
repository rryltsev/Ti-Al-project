---
license: cc-by-nc-4.0
---
# DPA4-OMat24

DPA4-OMat24 provides a series of **DPA4 / SeZM** machine-learning
interatomic potentials trained on the **OMat24** dataset. The models are
SO(3)-equivariant and conservative, and span the full periodic table. They are
intended for inorganic materials within the chemical space covered by OMat24.

# Overview

General information:

| Item            | Value                                      |
| --------------- | ------------------------------------------ |
| Architecture    | DPA4 / SeZM                                |
| Backend         | PyTorch only                               |
| Precision       | float32                                    |
| Elements        | Full periodic table (H–Og)                 |
| Cutoff radius   | 6.0 Å                                      |
| Training data   | OMat24 (100,568,576 frames)                |
| Validation data | OMat24 (1,074,643 frames)                  |
| Trained with    | DeePMD-kit 3.2.0, PyTorch 2.11 / CUDA 12.8 |

Model variants:

| Model     | Parameters        |
| --------- | ----------------- |
| DPA4-Nano | 480,650 (0.5 M)   |
| DPA4-Mini | 655,504 (0.7 M)   |
| DPA4-Neo  | 1,125,372 (1.1 M) |
| DPA4-Air  | 5,148,611 (5.1 M) |
| DPA4-Plus | 8,849,376 (8.8 M) |

Recommended sizes:

Mini and Neo are recommended for general use. Use Air or Plus when
higher accuracy is required.

Released files:

A release is identified by a date-based `<version>`
carried in the file name, so additional sizes or re-trained checkpoints of an
existing size can be published alongside earlier ones. Each release provides,
for every variant, a checkpoint and its training configuration:

| File                                   | Description             |
| -------------------------------------- | ----------------------- |
| `DPA4-<Variant>-OMat24-<version>.pt`   | Model checkpoint.       |
| `DPA4-<Variant>-OMat24-<version>.json` | Training configuration. |

`<Variant>` is one of the model variants listed above.

The models predict atomic energies, from which forces and virials are obtained
by differentiation:

$\qquad \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}$

The descriptor propagates vector and higher-order angular (SO(3)-equivariant)
features through message passing and reads out only the final `l = 0` scalar to
the energy fitting network. The architecture is described in the
[DeePMD-kit DPA4 documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html).

# Model architecture

Names in parentheses are the corresponding DeePMD-kit configuration keys. The
complete configurations, including the full `type_map`, are given in the
released input files.

**Size-dependent parameters**

| Parameter                               | DPA4-Nano       | DPA4-Mini        | DPA4-Neo          | DPA4-Air          | DPA4-Plus         | Description                                                                                                                |
| --------------------------------------- | --------------- | ---------------- | ----------------- | ----------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Feature channels (`channels`)           | 32              | 32               | 32                | 64                | 64                | Total channels per (l, m) coefficient.                                                                                     |
| Focus streams (`n_focus`)               | 1               | 1                | 2                 | 1                 | 1                 | Parallel focus streams inside the SO(2) convolution.                                                                       |
| Interaction blocks (`n_blocks`)         | 2               | 2                | 2                 | 3                 | 4                 | Number of message-passing blocks.                                                                                          |
| SO(2) mixing layers (`mixing_layers`)   | 3               | 3                | 3                 | 4                 | 4                 | Number of SO(2) mixing layers per block.                                                                                   |
| Max angular degree (`lmax`)             | 1               | 2                | 3                 | 3                 | 4                 | Maximum degree of the equivariant representation.                                                                          |
| Radial degree mixer (`radial_so2_mode`) | `none`          | `degree_channel` | `degree_channel`  | `degree_channel`  | `degree_channel`  | Dynamic radial degree-mixer mode in the SO(2) convolution; `degree_channel` uses a per-channel cross-degree radial kernel. |
| Mixer rank (`radial_so2_rank`)          | –               | 1                | 1                 | 1                 | 2                 | Low-rank factorization rank for the `degree_channel` kernel (0 = full per-channel kernel).                                 |
| Total parameters                        | 480,650 (0.5 M) | 655,504 (0.7 M)  | 1,125,372 (1.1 M) | 5,148,611 (5.1 M) | 8,849,376 (8.8 M) | Trainable parameter count.                                                                                                 |

**Shared settings**

The following settings are identical across all variants.

| Parameter                                   | Value                      | Description                                                                                                    |
| ------------------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Environment seeding (`use_env_seed`)        | True                       | Seed the initial node state with local-environment information via environment-matrix FiLM conditioning.       |
| FFN blocks (`ffn_blocks`)                   | 1                          | Number of FFN sublayers per interaction block.                                                                 |
| Radial basis (`basis_type`)                 | `bessel`                   | Radial basis type (`bessel` or `gaussian`).                                                                    |
| Radial functions (`n_radial`)               | 16                         | Number of radial basis functions.                                                                              |
| Max SO(2) order (`mmax`)                    | 1                          | Maximum SO(2) order.                                                                                           |
| Attention heads (`n_atten_head`)            | 1                          | Heads aggregating messages in the SO(2) convolution (0 = envelope-weighted scatter-sum).                       |
| SO(3) message branch (`message_node_so3`)   | True                       | Enable the post-aggregation SO(3) Wigner-D grid-net branch (message as query, node state as context).          |
| Normalization placement (`sandwich_norm`)   | [false, true, true, false] | Pre/post-norm switches for residual branches, ordered `[so2_pre, so2_post, ffn_pre, ffn_post]`.                |
| Activation (`activation_function`)          | `silu`                     | Base activation for helper MLPs, the SO(2) gated path, and the output FFN.                                     |
| GLU (`glu_activation`)                      | True                       | Base GLU switch for the FFN (e.g. SiLU → SwiGLU).                                                              |
| FFN width (`ffn_neurons`)                   | 0 (auto)                   | Hidden width for block and output FFNs; `0` resolves the width from `channels`.                                |
| FFN SO(3) grid (`ffn_so3_grid`)             | True                       | Use the Wigner-D SO(3) grid in the block-internal FFN.                                                         |
| Grid-branch routing (`grid_branch`)         | [0, 0, 1]                  | Scalar-routed polynomial product branches per grid path, ordered `[node_wise, message_node, ffn]`.             |
| SO(3) read-out (`so3_readout`)              | `mlp`                      | Read-out mode for the final l=0 descriptor; `mlp` folds l>0 geometry into l=0 via a point-wise SO(3) grid MLP. |
| Fitting hidden width (`fitting_net.neuron`) | [0] (auto)                 | Fitting-net hidden-layer widths; `0` is an auto-width placeholder resolved from the descriptor width.          |
| Cut-off radius (`rcut`)                     | 6.0 Å                      | Neighbor cut-off radius.                                                                                       |

**Precision and compilation**

| Parameter                       | Value   | Description                                                                                 |
| ------------------------------- | ------- | ------------------------------------------------------------------------------------------- |
| Compute precision (`precision`) | float32 | Precision of the descriptor parameters.                                                     |
| Mixed precision (`use_amp`)     | True    | Automatic mixed precision with bfloat16 on CUDA during training.                            |
| TF32 matmul (`enable_tf32`)     | True    | TF32 matmul precision for CUDA training forwards (eval-time TF32 is controlled separately). |
| Compiled path (`use_compile`)   | True    | Use the compiled DPA4/SeZM training path.                                                   |

# Validation

Results on the OMat24 validation set, with published baselines for reference.
For each DPA4 size, results are listed per release `<version>` and
evaluated over the full validation set (1,074,643 frames) as the
frame-count weighted average. Relative to earlier releases, the
`v20260805` checkpoints have improved PES smoothness.

| Model                | Version   | Predictionᵃ | Energyᵇ | Forceᵇ | Stressᵇ | Paramsᶜ    | Training hoursᵈ |
| -------------------- | --------- | ----------- | ------- | ------ | ------- | ---------- | --------------- |
| MACE-omat-0-S        | –         | Gradient    | 17.9    | 85.9   | 3.5     | 8.2M       | –               |
| MACE-omat-0-M        | –         | Gradient    | 16.3    | 78.4   | 3.3     | 9.1M       | –               |
| EquiformerV2-S       | –         | Direct      | 11.0    | 49.2   | 2.4     | 31M        | –               |
| EquiformerV2-M       | –         | Direct      | 10.0    | 44.8   | 2.3     | 87M        | –               |
| EquiformerV2-L       | –         | Direct      | 9.6     | 43.1   | 2.3     | 154M       | –               |
| eSEN                 | –         | Gradient    | 10.7    | 47.3   | 2.6     | 30M        | –               |
| UMA-S                | –         | Direct+Grad | 11.3    | 57.1   | 2.9     | 150M (6M)  | 46,080ᵉ         |
| UMA-M                | –         | Direct+Grad | 10.0    | 47.3   | 2.7     | 1.4B (50M) | 129,024ᵉ        |
| UMA-L                | –         | Direct      | 9.7     | 43.5   | 2.5     | 700M       | 95,232ᵉ         |
| EquiformerV3, Lₘₐₓ=4 | –         | Direct+Grad | 10.4    | 43.5   | 2.6     | 30M        | 4,782           |
| EquiformerV3, Lₘₐₓ=6 | –         | Direct+Grad | 10.1    | 41.6   | 2.5     | 49M        | 9,197           |
| **DPA4 series**      |           |             |         |        |         |            |                 |
| **DPA4-Nano**        | v20260805 | Gradient    | 18.7    | 96.1   | 3.4     | 0.5M       | –               |
| **DPA4-Mini**        | v20260704 | Gradient    | 14.0    | 70.7   | 2.9     | 0.7M       | 79              |
| **DPA4-Mini**        | v20260805 | Gradient    | 14.0    | 70.3   | 2.9     | 0.7M       | –               |
| **DPA4-Neo**         | v20260704 | Gradient    | 12.3    | 58.7   | 2.7     | 1.1M       | 139             |
| **DPA4-Neo**         | v20260805 | Gradient    | 12.1    | 58.7   | 2.7     | 1.1M       | –               |
| **DPA4-Air**         | v20260704 | Gradient    | 10.6    | 51.0   | 2.6     | 5.1M       | 205             |
| **DPA4-Air**         | v20260805 | Gradient    | 10.7    | 52.2   | 2.6     | 5.1M       | –               |
| **DPA4-Plus**        | v20260704 | Gradient    | 9.8     | 45.6   | 2.5     | 8.8M       | 336             |
| **DPA4-Plus**        | v20260805 | Gradient    | 10.0    | 47.0   | 2.5     | 8.8M       | –               |

- ᵃ Direct+Grad denotes direct-force pre-training followed by gradient fine-tuning.
- ᵇ Mean absolute errors: meV per atom for energy, meV/Å for forces, meV/Å³ for stress; lower is better.
- ᶜ Total parameters; parentheses give active parameters for UMA MoLE models.
- ᵈ Training hours are reported as equivalent H100 GPU-hours; DPA4 H20 training runs are converted using theoretical FP32 peak throughput.
- ᵉ UMA costs include multitask pre-training and fine-tuning, and are not OMat24-only training costs.

# Usage

DPA4/SeZM is a PyTorch-only model and uses the `.pt2` (AOTInductor) export path;
the TorchScript freeze path does not apply. Usage follows the
[DeePMD-kit DPA4 documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa4.html).
In the commands below, replace `<version>` with the release date and
`<Variant>` with the model variant.

## Installation

DPA4/SeZM is available in the DeePMD-kit main branch (version 3.2.0). Install
the latest code together with PyTorch 2.11 and CUDA 12.8:

```bash
pip install torch
pip install git+https://github.com/deepmodeling/deepmd-kit
```

For offline or backend-specific installation, see the
[DeePMD-kit installation guide](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/index.html).
The compiled inference path (`DP_COMPILE_INFER`) additionally requires
`torch >= 2.11` and CUDA `>= 12.6`.

## Evaluate a checkpoint

Run `dp test` on a released checkpoint with data in the standard or mixed-type
[DeePMD-kit data format](https://docs.deepmodeling.com/projects/deepmd/en/latest/data/system.html).
Replace `<Variant>` with the selected model variant:

```bash
dp --pt test -m DPA4-<Variant>-OMat24-<version>.pt -s /path/to/test/system -n 1000
```

## Freeze to `.pt2` for deployment

Freeze with `DP_TRITON_INFER=2`. This is the recommended setting: it enables
fused Triton kernels while keeping full float32 accumulation.

```bash
DP_TRITON_INFER=2 dp --pt freeze -c DPA4-<Variant>-OMat24-<version>.pt -o frozen_model
```

The PyTorch backend detects DPA4/SeZM and writes `frozen_model.pt2`. The `.pt2`
is an AOTInductor archive and is target-specific: it depends on the host
CPU/GPU, the GPU compute capability, and the libtorch version. Freeze on the
target machine rather than reusing a `.pt2` across different hardware.

**Inference precision is fixed at freeze time.** The `.pt2` is a forward-only
AOTInductor package, so the SO(2)-rotation kernel and matmul precision are
compiled into the graph at export and are not re-evaluated when the file is
later loaded by ASE or LAMMPS. Set the relevant environment variables before
running `dp --pt freeze`:

| Variable          | Default       | Effect                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DP_TF32_INFER`   | `0` (highest) | float32 matmul precision: `0` highest, `1` high, `2` medium. Keep `0` for MD and other PES-smoothness-sensitive workflows.                                                                                                                                                                                                                                                                                             |
| `DP_TRITON_INFER` | `0`           | Recommended freeze setting: `2`. Opt-in fused Triton inference kernels (CUDA), cumulative: `0` off; `1` universal kernels; `2` adds table-tuned SO(2) value-path kernels; `3` adds fp16 tensor-core mixing GEMMs. Levels `0`–`2` keep full float32 accumulation; `3`'s accuracy impact is negligible (it does not affect even second-order PES smoothness) and gives a large speedup on GPUs with strong tensor cores. |

Accepted boolean values: `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off`.

## Run in LAMMPS

The frozen `.pt2` is used through `pair_style deepmd`. A minimal setup:

```lammps
units           metal
atom_style      atomic
atom_modify     map yes

neighbor        2.0 bin
read_data       system.lmp

pair_style      deepmd frozen_model.pt2
pair_coeff      * * O H
```

`atom_modify map yes` is required: `.pt2` graph inference relies on an explicit
ghost/periodic-image to local-atom map, and the model fails fast if the map is
absent. The element names after `pair_coeff * *` bind LAMMPS atom types to
entries of the model's `type_map` in order (here types 1 and 2 to `O` and `H`);
if omitted, the mapping falls back to the `type_map` stored in the `.pt2`
metadata.

Multi-GPU (MPI) inference uses the same `.pt2`. Launch one MPI rank per GPU and
make every target device visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 lmp -in in.lammps
```

## Run with ASE

```python
from ase.io import read
from deepmd.calculator import DP

atoms = read("structure.cif")
atoms.calc = DP(model="frozen_model.pt2")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

## Adjust the energy bias for a new system

DFT energy labels differ between datasets by an arbitrary per-element constant.
Before evaluating or simulating a system whose energy reference differs from
OMat24, the per-element energy bias can be refit to that system without
retraining any network weights, which often improves energy accuracy:

```bash
dp --pt change-bias DPA4-<Variant>-OMat24-<version>.pt -s /path/to/system
```

This updates only the energy shift and writes an adjusted checkpoint; the
descriptor and fitting-net weights are unchanged.

## Fine-tune on a downstream dataset

The released checkpoints serve as pretrained initializations. Start from the
corresponding released input file and keep the entire `model` section unchanged —
descriptor, fitting net, and the full-periodic-table `type_map`, which the type
embeddings are indexed by and must not be changed. Replace only the
training/validation data with the downstream dataset and use a small learning
rate (e.g. `start_lr = 1e-4`):

```bash
dp --pt train input_finetune.json --finetune DPA4-<Variant>-OMat24-<version>.pt
```

DPA4/SeZM also supports LoRA adapters for single-task fine-tuning; the best
checkpoints fold the LoRA deltas back into the base weights, producing a plain
DPA4/SeZM checkpoint suitable for deployment.

# Intended use and limitations

- Intended for energy, force, and stress prediction of inorganic materials
  within the chemistry and configuration space covered by OMat24. Accuracy
  outside this domain (e.g. molecules, surfaces with adsorbates, far
  off-equilibrium structures) is not guaranteed and should be validated before
  use.
- PyTorch backend only; export is `.pt2` (AOTInductor), and model compression
  is not supported.
- Multi-GPU (MPI) LAMMPS inference applies to the plain energy model; ZBL zone
  bridging and spin models run on a single MPI rank.

# Citation

If you use these models, please cite the DPA4 paper and DeePMD-kit, and
acknowledge the OMat24 dataset.

```bibtex
@article{li2026dpa4,
  title   = {{DPA4}: Pushing the Accuracy-Cost Frontier of Interatomic
             Potentials with {EMFA} {SO(2)} Convolution},
  author  = {Li, Tiancheng and Li, Wentao and Peng, Anyang and Xue, Jianming
             and Zhang, Linfeng and Zhang, Duo and Wang, Han},
  journal = {arXiv preprint arXiv:2606.02419},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.02419},
  url     = {https://arxiv.org/abs/2606.02419}
}

@article{Wang_ComputPhysCommun_2018_v228_p178,
  author    = {Wang, Han and Zhang, Linfeng and Han, Jiequn and E, Weinan},
  title     = {{DeePMD-kit: A deep learning package for many-body potential
               energy representation and molecular dynamics}},
  journal   = {Comput. Phys. Comm.},
  volume    = {228},
  pages     = {178--184},
  year      = {2018},
  doi       = {10.1016/j.cpc.2018.03.016}
}

@article{Zeng_JChemPhys_2023_v159_p054801,
  author    = {Jinzhe Zeng and Duo Zhang and Denghui Lu and Pinghui Mo and Zeyu
               Li and Yixiao Chen and Mari{\'a}n Rynik and Li'ang Huang and Ziyao
               Li and Shaochen Shi and Yingze Wang and Haotian Ye and Ping Tuo
               and Jiabin Yang and Ye Ding and Yifan Li and Davide Tisi and Qiyu
               Zeng and Han Bao and Yu Xia and Jiameng Huang and Koki Muraoka and
               Yibo Wang and Junhan Chang and Fengbo Yuan and Sigbj{\o}rn
               L{\o}land Bore and Chun Cai and Yinnian Lin and Bo Wang and Jiayan
               Xu and Jia-Xin Zhu and Chenxing Luo and Yuzhi Zhang and Rhys E A
               Goodall and Wenshuo Liang and Anurag Kumar Singh and Sikai Yao and
               Jingchao Zhang and Renata Wentzcovitch and Jiequn Han and Jie Liu
               and Weile Jia and Darrin M York and Weinan E and Roberto Car and
               Linfeng Zhang and Han Wang},
  title     = {{DeePMD-kit v2: A software package for deep potential models}},
  journal   = {J. Chem. Phys.},
  volume    = {159},
  issue     = {5},
  pages     = {054801},
  year      = {2023},
  doi       = {10.1063/5.0155600}
}

@article{Zeng_JChemTheoryComput_2025_v21_p4375,
  author    = {Jinzhe Zeng and Duo Zhang and Anyang Peng and Xiangyu Zhang and
               Sensen He and Yan Wang and Xinzijian Liu and Hangrui Bi and Yifan
               Li and Chun Cai and Chengqian Zhang and Yiming Du and Jia-Xin Zhu
               and Pinghui Mo and Zhengtao Huang and Qiyu Zeng and Shaochen Shi
               and Xuejian Qin and Zhaoxi Yu and Chenxing Luo and Ye Ding and
               Yun-Pei Liu and Ruosong Shi and Zhenyu Wang and Sigbj{\o}rn
               L{\o}land Bore and Junhan Chang and Zhe Deng and Zhaohan Ding and
               Siyuan Han and Wanrun Jiang and Guolin Ke and Zhaoqing Liu and
               Denghui Lu and Koki Muraoka and Hananeh Oliaei and Anurag Kumar
               Singh and Haohui Que and Weihong Xu and Zhangmancang Xu and
               Yong-Bin Zhuang and Jiayu Dai and Timothy J. Giese and Weile Jia
               and Ben Xu and Darrin M. York and Linfeng Zhang and Han Wang},
  title     = {{DeePMD-kit v3: A Multiple-Backend Framework for Machine Learning
               Potentials}},
  journal   = {J. Chem. Theory Comput.},
  volume    = {21},
  number    = {9},
  pages     = {4375--4385},
  year      = {2025},
  doi       = {10.1021/acs.jctc.5c00340}
}

@article{BarrosoLuque_NatComputSci_2026_v6_p642,
  author    = {Barroso-Luque, Luis and Shuaibi, Muhammed and Fu, Xiang and
               Wood, Brandon M. and Dzamba, Misko and Gao, Meng and
               Rizvi, Ammar and Zitnick, C. Lawrence and Ulissi, Zachary W.},
  title     = {The Open Materials 2024 ({OMat24}) inorganic materials dataset
               and models},
  journal   = {Nat. Comput. Sci.},
  volume    = {6},
  pages     = {642--652},
  year      = {2026},
  doi       = {10.1038/s43588-026-00996-w},
  url       = {https://doi.org/10.1038/s43588-026-00996-w}
}
```
