<p align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXVodWNsM3Bia3duZGljZzRqMTI2MGFiZjlkZzBwcmhuaWxydjlpaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/AOSwwqVjNZlDO/giphy.gif" width="50%" alt="Matrix Structural Analysis Banner">
</p>

# Matrix Structural Analysis

### Matrix-based structural analysis for learning, modelling, and visualizing 2D structures

**Author:** Msc. Ing. Carlos Andrés Celi Sánchez  
**Semester:** February–July 2026

<p align="center">
  <a href="https://github.com/Normando1945/Repo_Maxtrix_Analisys/actions/workflows/python-check.yml">
    <img src="https://github.com/Normando1945/Repo_Maxtrix_Analisys/actions/workflows/python-check.yml/badge.svg" alt="Python Check">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10 or newer">
</p>

## Overview

This repository supports the teaching and learning of **Matrix Structural Analysis**. It combines progressive Jupyter notebooks with a reusable Python package so that students can move from element-level formulations to the assembly and visualization of small 2D structural systems.

<p align="center">
  <img src="examples/frame_deformed_shape.png" alt="Original and deformed 2D moment frame with displacement contour" width="100%">
</p>

<p align="center"><em>Original and deformed configuration of a 2D moment frame, including displacement contours and nodal rotations. Generated in <a href="examples/Ejemplo_Class_Matricial.ipynb">Ejemplo_Class_Matricial.ipynb</a>.</em></p>

The current material covers the main ideas required to work with matrix-based structural models:

- degrees of freedom, local and global coordinate systems;
- local stiffness matrices and coordinate transformations;
- 2D truss elements with axial deformation;
- 2D moment-frame elements with axial, flexural, and shear deformation;
- rigid-end offsets in frame elements;
- assembly using a location matrix;
- displacement and matrix visualization; and
- flexibility-method diagrams and worked numerical examples.

> This is an educational repository. The implementations are intended for study, verification, and small examples; they are not a replacement for production structural-analysis software or professional design verification.

## Repository structure

```text
Repo_Maxtrix_Analisys/
├── examples/
│   ├── Basic_Coding.ipynb
│   ├── Ejemplo_Class_Matricial.ipynb
│   ├── Example_Truss.ipynb
│   ├── Example_Truss_2.ipynb
│   ├── Flex_manual_AM_GR1_2026_01.ipynb
│   ├── frame_deformed_shape.png       # 2D frame visualization shown above
│   └── *.png                         # figures used by the notebooks
├── repo_maxtrix_analisys/
│   ├── __init__.py                   # public package interface
│   └── core.py                       # element, assembly, and plotting classes
├── .github/workflows/python-check.yml
├── README.md
├── requirements.txt
└── setup.py
```

## Learning path and examples

The notebooks are designed to be read and executed progressively.

| Notebook | Main purpose |
| --- | --- |
| `Basic_Coding.ipynb` | Introductory Python and numerical-matrix work used in the course. |
| `Ejemplo_Class_Matricial.ipynb` | Class-based formulation of 2D frame elements, stiffness matrices, transformations, and assembly. |
| `Example_Truss.ipynb` | Fundamental 2D truss-element example. |
| `Example_Truss_2.ipynb` | Additional truss analysis and application exercises. |
| `Flex_manual_AM_GR1_2026_01.ipynb` | Manual flexibility-method diagrams and comparison plots. |

Recommended order: start with **Basic Coding**, continue with **Ejemplo Class Matricial**, then study the two truss examples and the flexibility-method notebook.

## Package capabilities

The package is intentionally compact. Its public classes are exported from `repo_maxtrix_analisys`.

| Class | Purpose |
| --- | --- |
| `MF_K_T_L_Element2D` | Local stiffness and transformation matrix for a 2D moment-frame element with axial, flexural, shear, and rigid-end-offset effects. |
| `ARM_K_T_Element2D` | Local stiffness and transformation matrix for a 2D axial bar/truss element. |
| `Assembler` | Adds an element stiffness matrix into a structural matrix from a location matrix. |
| `Manager_K_T_elements2D` | Collects element stiffness and transformation matrices for frame or truss workflows. |
| `StiffnessMatrix_simple` | Introductory six-degree-of-freedom frame stiffness formulation. |
| `SimpleMatrixStack` | Small introductory matrix-generation example. |
| `M_visual_2D_3D` | Matrix visualizer. |
| `PlotGlobalDislplacemet` | Plotter for original and deformed 2D moment-frame geometry. |
| `Manual_Flexural_Method` | Plotter for flexibility-method action diagrams. |

## Requirements

- Python **3.10 or newer**
- Git
- Visual Studio Code (recommended)
- VS Code extensions: **Python** and **Jupyter**

The package uses NumPy, Pandas, and Matplotlib. Jupyter and IPython kernel support are needed to execute the course notebooks.

## Installation on Windows

Open Command Prompt or the VS Code integrated terminal, then run:

```bash
git clone https://github.com/Normando1945/Repo_Maxtrix_Analisys.git
cd Repo_Maxtrix_Analisys
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib jupyter ipykernel
pip install -e .
```

If you prefer to install from the repository list first, you can also run:

```bash
pip install -r requirements.txt
```

Then register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name matrix-analysis --display-name "Python (Matrix Analysis)"
```

In VS Code, select the `venv` interpreter and, when working with a notebook, select **Python (Matrix Analysis)** as the kernel.

## Quick start

The following example creates a 2D moment-frame element and obtains its local stiffness and transformation matrices.

```python
from repo_maxtrix_analisys import MF_K_T_L_Element2D

element = MF_K_T_L_Element2D(
    E=25e9,       # elastic modulus [Pa]
    A=0.12,       # area [m²]
    I=0.002,      # second moment of area [m⁴]
    L=5.0,        # element length [m]
    nu=0.20,      # Poisson ratio
    f=6 / 5,      # shear correction factor
    dA=0.0,       # rigid-end offset at A [m]
    dB=0.0,       # rigid-end offset at B [m]
    thetha=0.0,   # orientation angle [degrees]
)

k_local = element.stiffness_matrix_MF_EI_AE_GAf_da_db()
transformation = element.transformation_matrix_2D()

print(k_local)
print(transformation)
```

For complete structural examples, open and run the notebooks in the order suggested above. Execute cells step by step: the notebooks are deliberately explicit so that each operation can be inspected and discussed.

## Working with the repository

Before each class session, update your local clone:

```bash
cd Repo_Maxtrix_Analisys
venv\Scripts\activate
git pull
```

Students are encouraged to experiment in their own local copies or forks. Please keep the original course repository organized and report reproducible issues or suggestions through GitHub.

## Verification

The repository includes a GitHub Actions workflow that installs the package and verifies that it can be imported with Python 3.10. Locally, the same basic check is:

```bash
python -c "import repo_maxtrix_analisys; print('Package imported successfully')"
```

## Citation

If you use this repository in academic work, class projects, reports, or educational material, please cite it as follows.

```bibtex
@misc{celi2026matrix,
  author       = {Carlos Andrés Celi Sánchez},
  title        = {Matrix Structural Analysis: Matrix-Based Structural Analysis for 2D Structures},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Normando1945/Repo_Maxtrix_Analisys}}
}
```

Celi Sánchez, C. A. (2026). *Matrix Structural Analysis: Matrix-Based Structural Analysis for 2D Structures* [Structural Engineering]. GitHub. https://github.com/Normando1945/Repo_Maxtrix_Analisys

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

This is the official repository for the course. Students may use the material, report bugs, and propose improvements through issues or pull requests. Changes to the official course content remain under the supervision of the author.
