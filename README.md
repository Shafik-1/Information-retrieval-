# Information Retrieval Project

This project implements a **Positional Index** using Apache Spark (Part 1) and a **Search Engine** with Query Optimization and TF-IDF ranking (Part 2).

## Methodology

### Part 1: Indexing (MapReduce)

We utilize a **MapReduce** approach on Apache Spark to build the index:

1.  **Tokenize (Map):** Reads documents and maps every word to `(Term, (DocID, Position))`.
2.  **Sort (Shuffle):** Spark groups data so all occurrences of a term are together.
3.  **Merge (Reduce):** Collapses the list into a dictionary format: `Term -> DocID: [Pos1, Pos2...]`.

### Part 2: Search Engine Logic

1.  **Query Optimization ("Start Small"):** For `AND` queries, terms are sorted by **Document Frequency (DF)**. The engine processes the rarest terms first (smallest lists) to minimize intersection cost.
2.  **Phrase Queries:** Implements positional logic `pos(B) == pos(A) + 1` to ensure exact phrase matches.
3.  **Ranking:**
    - **TF-IDF:** Uses `(1 + log(tf)) * idf` weighting.
    - **Cosine Similarity:** Ranks results by the cosine angle between Query and Document vectors.
    - **Product Table:** Displays a detailed breakdown of `q_normalized * d_normalized` for every matching document.

## Prerequisites

1.  **Python 3.x**: Ensure Python is installed.
2.  **Java 8 or 11**: Required for Apache Spark.
3.  **Apache Spark**: Installed and configured with `SPARK_HOME`.

## Usage

### Part 1: Positional Index (Spark App)

Reads dataset, builds index, and saves to `output.txt`.

```bash
spark-submit 1_Positional_Index.py
```

- **Result:** Generates `output.txt`.

### Part 2: Search Engine

Parses `output.txt` and performs ranked searches.

```bash
python3 2_search_engine.py "fools fear in"
```

- **Detailed Output:** Prints TF, TF-IDF, IDF, and Normalized matrices.
- **Visual Format:** Matches the "Product Table" structure (Rows: Query Terms, Cols: Docs).
- **Precision:** All values displayed to **6 decimal places**.
- **Auto-Logging:** Automatically saves all terminal output to `response.txt`.
  - **Versioning:** If `response.txt` exists, it saves to `response1.txt`, `response2.txt`, etc.

**Supported Queries:**

- **Phrase:** `"fools fear in"`
- **Boolean AND:** `'antony' AND 'brutus'`
- **Boolean NOT:** `'angels' AND NOT 'fools'`

## Clean Code Structure

- **1_Positional_Index.py**: Minimalist Spark logic with clear `Step 1`, `Step 2` comments.
- **2_search_engine.py**: Professional, streamlined search logic without verbose comments.
