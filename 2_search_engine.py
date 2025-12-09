import math
import sys
import os
import re

OUTPUT_FILE = "output.txt"

def natural_sort_key(filename):
    """Sort filenames numerically (1.txt, 2.txt, ..., 10.txt)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def load_index(file_path):
    print(f"Loading index from {file_path}...")
    index = {}
    all_docs = set()
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None, None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('<') or not line.endswith('>'):
                continue
            
            # Remove < and >
            content = line[1:-1]
            
            # Split term and doc info
            # Format: term : doc1: pos1, pos2 ; doc2: pos1 ...
            parts = content.split(' : ', 1)
            if len(parts) != 2:
                continue
                
            term = parts[0]
            doc_info_str = parts[1]
            
            index[term] = {}
            
            # Split documents
            doc_entries = doc_info_str.split(' ; ')
            for entry in doc_entries:
                # doc: pos1, pos2
                doc_parts = entry.split(': ')
                if len(doc_parts) != 2:
                    continue
                
                doc_name = doc_parts[0]
                # positions might be comma separated
                positions = [int(p) for p in doc_parts[1].split(', ')]
                
                index[term][doc_name] = positions
                all_docs.add(doc_name)
                
    return index, sorted(list(all_docs), key=natural_sort_key)

def print_table(headers, data, title):
    print(f"\n--- {title} ---")
    if not data:
        print("No data.")
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))
            
    # Print header
    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in data:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)))

def get_phrase_docs(index, phrase):
    terms = phrase.lower().split()
    if not terms:
        return set()
    
    # Check if all terms exist
    for term in terms:
        if term not in index:
            return set()
            
    # Start with docs containing the first term
    candidate_docs = set(index[terms[0]].keys())
    
    matched_docs = set()
    for doc in candidate_docs:
        # Check if all terms are present in this doc
        present = True
        for term in terms:
            if doc not in index[term]:
                present = False
                break
        if not present:
            continue
            
        # Check positions for phrase
        # positions of term[i] must contain p such that term[i+1] has p+1
        current_positions = index[terms[0]][doc]
        
        for i in range(1, len(terms)):
            next_term = terms[i]
            next_positions = set(index[next_term][doc])
            
            new_positions = []
            for p in current_positions:
                if (p + 1) in next_positions:
                    new_positions.append(p + 1)
            
            current_positions = new_positions
            if not current_positions:
                break
        
        if current_positions:
            matched_docs.add(doc)
            
    return matched_docs

def parse_query(query):
    # Split by ' AND '
    # Handle 'NOT '
    # Case-insensitive split is harder, but let's assume user uses AND/NOT or we normalize
    # For now, let's normalize the operators by replacing ' and ' with ' AND ' etc if we wanted, 
    # but the requirement says "AND", "AND NOT".
    
    parts = query.split(' AND ')
    must_include = []
    must_exclude = []
    
    for part in parts:
        part = part.strip()
        if part.startswith('NOT '):
            phrase = part[4:].strip().strip('"\'')
            must_exclude.append(phrase)
        else:
            phrase = part.strip().strip('"\'')
            must_include.append(phrase)
            
    return must_include, must_exclude

def search(query, index, tf_idf_matrix, idf_dict, doc_norms):
    must_include, must_exclude = parse_query(query)
    
    print(f"DEBUG: Searching for: {must_include}, Excluding: {must_exclude}")
    
    # 1. Find docs for included phrases
    if not must_include:
        return []
        
    result_docs = None
    
    for phrase in must_include:
        docs = get_phrase_docs(index, phrase)
        if result_docs is None:
            result_docs = docs
        else:
            result_docs = result_docs.intersection(docs)
            
    if not result_docs:
        return []
        
    # 2. Exclude docs
    for phrase in must_exclude:
        docs = get_phrase_docs(index, phrase)
        result_docs = result_docs.difference(docs)
        
    # 3. Rank by Similarity (Cosine)
    # Query Vector Q: w_t,q = (1 + log(tf_q)) * idf_t
    
    # Collect all unique terms in query
    query_terms = []
    for phrase in must_include:
        query_terms.extend(phrase.lower().split())
        
    # Calculate Query Vector Weights
    query_tf = {}
    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1
    
    # First, calculate q_norm (magnitude of query vector)
    query_weights = {}
    q_norm_sq = 0
    
    for term, tf in query_tf.items():
        if term in idf_dict:
            idf = idf_dict[term]
            w = (1 + math.log10(tf)) * idf
            query_weights[term] = w
            q_norm_sq += w * w
            
    q_norm = math.sqrt(q_norm_sq)
    
    # Print query term analysis table
    print("\n--- Query Term Analysis ---")
    query_analysis_data = []
    for term in sorted(query_tf.keys()):
        tf_raw = query_tf[term]
        log_tf = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
        idf = idf_dict.get(term, 0)
        tf_idf = log_tf * idf
        normalized = tf_idf / q_norm if q_norm > 0 else 0
        
        query_analysis_data.append([
            term,
            tf_raw,
            f"{log_tf:.4f}",
            f"{idf:.4f}",
            f"{tf_idf:.4f}",
            f"{normalized:.4f}"
        ])
    
    if query_analysis_data:
        print(f"{'Term':<15} | {'TF-Raw':<10} | {'log(TF)+1':<12} | {'IDF':<10} | {'TF*IDF':<10} | {'Normalized':<12}")
        print("-" * 80)
        for row in query_analysis_data:
            print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<12} | {row[3]:<10} | {row[4]:<10} | {row[5]:<12}")
    
    # Print query length
    print(f"\nQuery Length: {q_norm:.6f}")
    
    scores = []
    print("\n--- Similarity Scores ---")
    for doc in result_docs:
        dot_product = 0
        d_norm = doc_norms.get(doc, 0)
        
        for term, q_w in query_weights.items():
            if term in tf_idf_matrix and doc in tf_idf_matrix[term]:
                d_w = tf_idf_matrix[term][doc]
                dot_product += q_w * d_w
        
        sim = 0
        if q_norm > 0 and d_norm > 0:
            sim = dot_product / (q_norm * d_norm)
        
        print(f"Similarity(q, {doc}) = {sim:.4f}")
        scores.append((doc, sim))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def main():
    index, all_docs = load_index(OUTPUT_FILE)
    if index is None:
        return

    # 1. Compute TF
    tf_matrix = {}
    print("\nComputing Term Frequency (TF)...")
    tf_headers = ["Term"] + all_docs
    tf_data = []
    sorted_terms = sorted(index.keys())
    for term in sorted_terms:
        row = [term]
        tf_matrix[term] = {}
        for doc in all_docs:
            count = len(index[term].get(doc, []))
            tf_matrix[term][doc] = count
            row.append(count)
        tf_data.append(row)
    print_table(tf_headers, tf_data, "Term Frequency (TF)")
    
    # 2.5. Compute (1 + log(TF))
    print("\nComputing (1 + log(TF))...")
    log_tf_data = []
    for term in sorted_terms:
        row = [term]
        for doc in all_docs:
            tf = tf_matrix[term][doc]
            log_tf_val = 1 + math.log10(tf) if tf > 0 else 0
            row.append(int(log_tf_val))
        log_tf_data.append(row)
    print_table(tf_headers, log_tf_data, "(1 + log(TF)) Matrix")

    # 2. Compute IDF
    # print("\nComputing IDF...")
    N = len(all_docs)
    idf_dict = {}
    idf_data = []
    for term in sorted_terms:
        df = len(index[term])
        idf = math.log10(N / df) if df > 0 else 0
        idf_dict[term] = idf
        idf_data.append([term, df, f"{idf:.4f}"])
    print_table(["Term", "DF", "IDF"], idf_data, "Document Frequency (DF) and Inverse Document Frequency (IDF)")

    # 3. Compute TF-IDF
    print("\nComputing TF-IDF Matrix...")
    tf_idf_matrix = {}
    tf_idf_data = []
    for term in sorted_terms:
        row = [term]
        tf_idf_matrix[term] = {}
        for doc in all_docs:
            tf = tf_matrix[term][doc]
            idf = idf_dict[term]
            val = tf * idf
            tf_idf_matrix[term][doc] = val
            row.append(f"{val:.4f}")
        tf_idf_data.append(row)
    print_table(tf_headers, tf_idf_data, "TF-IDF Matrix")

    # 4. Compute Doc Lengths (for Cosine Similarity)
    doc_norms = {}
    for doc in all_docs:
        norm_sq = 0
        for term in sorted_terms:
            val = tf_idf_matrix[term][doc]
            norm_sq += val * val
        doc_norms[doc] = math.sqrt(norm_sq)

    # print_table(["Term", "IDF"], idf_data, "Inverse Document Frequency (IDF)")

    # 4.5. Print Document Lengths
    print("\nComputing Document Lengths...")
    doc_length_data = []
    for doc in sorted(all_docs, key=natural_sort_key):
        length = doc_norms[doc]
        doc_length_data.append([doc, f"{length:.4f}"])
    print_table(["Document", "Length"], doc_length_data, "Document Lengths")

    # 3.5. Compute Normalized TF-IDF
    print("\nComputing Normalized TF-IDF Matrix...")
    normalized_tf_idf_data = []
    for term in sorted_terms:
        row = [term]
        for doc in all_docs:
            tf_idf_val = tf_idf_matrix[term][doc]
            d_norm = doc_norms[doc]
            normalized_val = tf_idf_val / d_norm if d_norm > 0 else 0
            row.append(f"{normalized_val:.4f}")
        normalized_tf_idf_data.append(row)
    print_table(tf_headers, normalized_tf_idf_data, "Normalized TF-IDF Matrix")

    # 5. Search Interface
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}")
        results = search(query, index, tf_idf_matrix, idf_dict, doc_norms)
    else:
        print("\n--- Search Engine Ready ---")
        while True:
            try:
                query = input("\nEnter query (e.g., 'angels fools' AND NOT 'mercy') or 'exit': ")
                if query.lower() == 'exit':
                    break
                results = search(query, index, tf_idf_matrix, idf_dict, doc_norms)
            except (EOFError, KeyboardInterrupt):
                break

if __name__ == "__main__":
    main()
