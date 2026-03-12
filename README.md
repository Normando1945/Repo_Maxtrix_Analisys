<p align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExcXVodWNsM3Bia3duZGljZzRqMTI2MGFiZjlkZzBwcmhuaWxydjlpaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/AOSwwqVjNZlDO/giphy.gif" width="50%" alt="Matrix Structural Analysis Banner">
</p>

# Matrix Structural Analysis

### Introduction to Matrix-Based Analysis of Structures

**Author:** Msc. Ing. Carlos Andrés Celi Sánchez  
**Semester:** FEB – 2026  

This repository has been created to support the teaching and learning process of the **Matrix Structural Analysis** course during the current academic semester. It will progressively include theoretical notes, Python codes, numerical examples, and class-based implementations, starting from the fundamental concepts of matrix structural analysis and advancing toward more refined formulations for 2D frame elements.

## Course Roadmap

This repository is expected to progressively cover topics such as:

- Fundamental concepts of matrix structural analysis
- Degrees of freedom and coordinate systems
- Element local stiffness matrices
- Coordinate transformation
- Global assembly procedures
- Boundary conditions and system solution
- Internal forces and displacement recovery
- Class-based Python implementation
- Advanced 2D frame elements including axial, flexural, and shear deformations
- Rigid end offsets in matrix-based formulations

## Current Contents

At its current stage, the repository includes:

- An initial Python package for matrix-based structural analysis
- A class-based implementation for 2D moment-frame elements
- An example notebook introducing the element stiffness formulation
- Setup and dependency files for package installation

## Repository Structure

    Repo_Maxtrix_Analisys/
    │── examples/
    │   └── Ejemplo_Class_Matricial.ipynb
    │── repo_maxtrix_analisys/
    │   ├── __init__.py
    │   └── core.py
    │── README.md
    │── requirements.txt
    │── setup.py

## Installation

    git clone https://github.com/Normando1945/Repo_Maxtrix_Analisys.git
    cd Repo_Maxtrix_Analisys
    pip install -r requirements.txt
    pip install -e .

## First Example

The repository currently includes an introductory notebook:

- `examples/Ejemplo_Class_Matricial.ipynb`

This notebook presents the matrix-based formulation of a 2D moment-frame element and its implementation in Python using a class-based approach.