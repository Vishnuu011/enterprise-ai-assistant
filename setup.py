from setuptools import setup, find_packages
from pathlib import Path

# def read_requirements():
#     req_file = Path(__file__).parent / "requirements.txt"
#     if req_file.exists():
#         return req_file.read_text().splitlines()
#     return []

setup(
    name="enterprise-ai-assistant",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "streamlit",
        "numpy",
        "matplotlib",
        "seaborn",
        "groq",
        "pydantic",
        "ipython",
        "scikit-learn",
        "tqdm",
        "langchain-community",
        "langchain-core",
        "langchain-groq",
        "langgraph", 
        "pypdf", 
        "chromadb",
        "unstructured",
        "python-docx",
        "langchain-experimental",  
        "db-sqlite3",
        "pandas",
        "numpy",
        "sentence-transformers",
        "tabulate"
    ],
    author="Vishnu",
    author_email="vishnurrajeev@gmail.com",
    description="Enterprise Multi-Agent AI Assistant with Streamlit UI",
    long_description=Path(__file__).parent.joinpath("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Vishnuu011/enterprise-ai-assistant",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "enterprise-ai-assistant=app:main",  # CLI points to wrapper in app_ui.py
        ],
    },
)