# Automatic Essay Scoring Project

## Introduction
This project aims to develop an automatic essay scoring system using machine learning techniques. The system is designed to evaluate essays based on various linguistic, structural, and content-related features, providing a quantitative assessment of writing quality.

## Project Structure
The project directory contains the following files and folders:

- `AES_Pipeline_code.ipynb`: Jupyter Notebook containing the project code.
- `Report.pdf` : PDF file containing a brief summarized report about what has been done.
- `AES_Pipeline_Functions.py`: Python file with function definitions used in the notebook.
- `requirements_AES.txt`: Text file specifying the Python package requirements for the project.
- `JavaSetup8u391.exe`: Executable installer for Java.
- `essay_descriptions`: Folder containing Word documents with essay descriptions for each essay set and scoring guidelines.
- `dataset_summary.xlsx`: Excel file providing an overview of the dataset.
- `project_description.pdf`: PDF document describing the project in detail.
- `insights_and_examples`: Folder containing insights and examples for training and validation data decomposition.

## Requirements
To run the project, you will need Python IDE, Anaconda installed on your system. Additionally, ensure that the following Python packages are installed:

- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn
- spaCy
- etc...

You can install these packages using the provided `requirements.txt` file by running the following command in your terminal:
```bash
pip install -r requirements_AES.txt
```

## Java Installation

Java is required for certain components of the project. If you haven't already, download and install Java from the official website: [https://www.java.com/download/ie_manual.jsp].

<font color='red'><h3>| Attention:</h3></font>
- Don't forget to **add the path to your Java installation** to the system **PATH** environment variable.
- .exe Installer
Additionally, an executable installer is provided for Java. Download and run the installer to install Java on your system.

## Usage
1. Clone the repository to your local machine:
```bash
git clone https://github.com/your_username/automatic-essay-scoring.git
````

2. Navigate to the project directory:
```bash
cd automatic-essay-scoring
```

3. Launch Jupyter Notebook and open the AES_Pipeline_code.ipynb notebook:
```bash
jupyter notebook AES_Pipeline_code.ipynb
```

Finally, follow the instructions provided in the notebook to execute the code cells for data preprocessing, model training, and evaluation.
