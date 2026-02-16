from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt safely
def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        return [
            line.strip()
            for line in req_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []

setup(
    name="repo_maxtrix_analisys",
    version="0.1.0",
    author="Carlos Celi",
    description="Matrix-based structural analysis tools for 2D frame elements.",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.10",
)
