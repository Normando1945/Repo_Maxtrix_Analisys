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

## Prerequisites

Before working with this repository, students should make sure that the following software is installed on their computers:

- **Python 3.10 or newer**
- **Git**
- **Visual Studio Code**
- **Python extension for VS Code**
- **Jupyter extension for VS Code**

These tools are necessary to clone the repository, create the Python environment, open the project correctly in Visual Studio Code, and run both Python scripts and notebooks.

## Installation Guide for Windows and VS Code

This section explains how to correctly install and run the repository on **Windows** using **Visual Studio Code**.

### Step 1. Open the Windows terminal

Before doing anything else, students should first open a standard **Windows terminal**.

They may use any of the following:

- **Command Prompt**
- **Windows PowerShell**

For this course, the recommended option is:

> **Command Prompt**

This helps avoid confusion with terminal commands, file paths, and virtual environment activation steps.

### Step 2. Clone the repository

Once the Windows terminal is open, run:

    git clone https://github.com/Normando1945/Repo_Maxtrix_Analisys.git

This command will download the repository to the current folder.

### Step 3. Move into the repository folder

After cloning the repository, enter the project folder with:

    cd Repo_Maxtrix_Analisys

From this point on, all commands should be executed inside this folder.

### Step 4. Open the repository in Visual Studio Code

Now that the repository already exists on the computer, open it in **Visual Studio Code** by running:

    code .

If this command does not work, students can simply open **Visual Studio Code** manually and then select the cloned repository folder.

### Step 5. Open the integrated terminal in VS Code

Once the repository has been opened in VS Code, it is recommended that students continue working from the **integrated terminal** inside VS Code.

To open the terminal in VS Code:

- Press **Ctrl + Shift + `**
- Or go to the top menu and select:  
  **Terminal > New Terminal**

A terminal panel will appear at the bottom of Visual Studio Code.

### Step 6. Verify that the terminal is Command Prompt

Inside VS Code, verify that the selected terminal is:

- **Command Prompt**

If another terminal appears and students want to change it:

1. Click the dropdown menu in the terminal panel.
2. Select **Command Prompt**.
3. Open a new terminal.

From this point on, it is recommended that all commands be executed from this terminal in VS Code.

### Step 7. Create a virtual environment

It is strongly recommended to create a virtual environment so that all students work with the same isolated Python setup.

Run:

    python -m venv venv

This command will create a folder called `venv` inside the repository.

### Step 8. Activate the virtual environment in Windows

If students are using **Command Prompt**, run:

    venv\Scripts\activate

After activation, `(venv)` should appear at the beginning of the terminal line. This indicates that the virtual environment is active.

### Step 9. Install the required dependencies

Once the virtual environment has been activated, install the required Python libraries with:

    pip install -r requirements.txt

This step installs all the packages needed by the repository.

### Step 10. Install the repository in editable mode

To allow Python to recognize the package correctly while developing and testing the code, run:

    pip install -e .

This is useful because the package can be modified during the semester without reinstalling it every time.

### Step 11. Install Jupyter support inside the environment

If students are going to work with notebooks in VS Code, it is recommended to also install `ipykernel`:

    pip install ipykernel

Then register the environment as a Jupyter kernel:

    python -m ipykernel install --user --name=venv --display-name "Python (Matrix Analysis)"

This will allow students to select the correct Python environment when opening notebooks.

### Step 12. Select the correct interpreter in VS Code

Inside **Visual Studio Code**, follow these steps:

1. Press **Ctrl + Shift + P**.
2. Search for: `Python: Select Interpreter`
3. Choose the interpreter corresponding to the `venv` environment.

If a notebook is opened, also make sure that the selected kernel is:

`Python (Matrix Analysis)`

### Step 13. Verify that the installation works correctly

A simple way to verify the installation is to open Python and try importing the main package.

Run:

    python

Then type:

    import repo_maxtrix_analisys
    print("Package imported successfully")

If no error appears, the installation was completed correctly.

## First Example

The repository currently includes an introductory notebook:

- `examples/Ejemplo_Class_Matricial.ipynb`

This notebook presents the matrix-based formulation of a 2D moment-frame element and its implementation in Python using a class-based approach.

## Recommended Workflow for Students

For each class session, students are encouraged to follow the workflow below:

1. Open the repository folder in VS Code.
2. Open the integrated Command Prompt terminal.
3. Activate the virtual environment.
4. Verify that the correct Python interpreter has been selected.
5. Open the corresponding notebook or Python file.
6. Run the examples step by step.
7. Modify the examples progressively as discussed in class.
8. Save the updated work in an organized manner.

This workflow helps maintain consistency during the semester and reduces the most common installation and execution errors.

## Updating the Repository

Since the repository will be updated progressively during the semester, students should regularly download the latest changes from GitHub.

### Step 1. Open the terminal

Open **Command Prompt** or the **integrated terminal in VS Code**.

### Step 2. Move into the repository folder

    cd Repo_Maxtrix_Analisys

### Step 3. Pull the latest changes

    git pull

This command downloads and merges the most recent changes from the remote repository into the local copy.

### Recommendation

Students are encouraged to run `git pull` before starting each class session in order to work with the latest version of the repository.

## Additional Notes

- If Git is not recognized in the terminal, it must be installed and added correctly to the system path.
- If Python is not recognized in the terminal, verify that Python was installed correctly and added to the system path.
- If a notebook does not run, first verify that the correct Python interpreter and Jupyter kernel have been selected.
- It is recommended that all package installations be done only after activating the virtual environment.
- Students should avoid installing packages globally unless it is absolutely necessary.

### Summary of the main installation and update commands

    git clone https://github.com/Normando1945/Repo_Maxtrix_Analisys.git
    cd Repo_Maxtrix_Analisys
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    pip install -e .

### Summary of the main update commands

    cd Repo_Maxtrix_Analisys
    venv\Scripts\activate
    git pull

## General Recommendation

Students are encouraged to keep this repository updated throughout the semester and use it as the main reference point for class examples, numerical implementations, and progressive development of matrix-based structural analysis tools in Python.

## Important Note for Students

This repository is maintained exclusively by the course author.

Students are expected to clone the repository and update their local copies during the semester. They should not modify the original online repository.

If students wish to experiment with the code, they are encouraged to do so in their local copies or in personal forks of the repository.