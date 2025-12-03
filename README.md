# Information Retrieval Project

This project implements a Positional Index using Apache Spark (Part 1) and a Search Engine with .TF-IDF ranking (Part 2).

## Prerequisites

1. **Python 3.x**: Ensure Python is installed.
2. **Java 8 or 11**: Required for Apache Spark.
   - **Linux**: `sudo apt install openjdk-11-jdk`
   - **Windows**: Download and install JDK 11 from Oracle or OpenJDK. Set `JAVA_HOME` environment variable.
3. **Apache Spark**:
   - **Linux**: Download from [spark.apache.org](https://spark.apache.org/downloads.html), extract, and set `SPARK_HOME` and add `bin` to `PATH`.
   - **Windows**:
     - Download Spark and extract it (e.g., to `C:\Spark`).
     - Set `SPARK_HOME` environment variable to your Spark folder.
     - Add `%SPARK_HOME%\bin` to your `Path`.
     - **Winutils**: You may need `winutils.exe` for Hadoop on Windows. Download it and place it in `%SPARK_HOME%\bin` or a separate `hadoop\bin` folder (set `HADOOP_HOME` accordingly).

## Installation

1. **Clone/Download** the project to your local machine.
2. **Dependencies**:
   - **Part 1 (Spark App)**: Relies on your global Apache Spark installation. No `pip install` required if using `spark-submit`.
   - **Part 2 (Search Engine)**: Uses standard Python libraries (`math`, `sys`, `os`). No external dependencies.

## Usage

### Part 1: Positional Index (Spark App)

This script reads the dataset, builds a positional index, and saves it to `output.txt`.

**Command:**

```bash
spark-submit ir.py
```

- **Note**: Ensure the `input_path` in `ir.py` points to your dataset folder.
  - **Windows Users**: Update this path to your local path, e.g., `file:///C:/Users/YourName/Project/dataset/*.txt`.

**Output:**

- Generates `output.txt` containing the positional index.

### Part 2: Search Engine

This script parses `output.txt`, computes TF-IDF, and allows you to search the documents.

**Command:**

```bash
python search_engine.py 'antony' AND 'brutus'
```

or 

```bash
python3 search_engine.py 'antony' AND 'brutus'
```

**Search Queries:**

- **Phrase Search**: `"angels fools"`
- **Boolean AND**: `'antony' AND 'brutus'`
- **Boolean AND NOT**: `'angels' AND NOT 'fools'`

**Example:**

```bash
python search_engine.py 'antony' AND 'brutus'
```
