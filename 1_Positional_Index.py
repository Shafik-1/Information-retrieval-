from pyspark import SparkContext, SparkConf
import os
import shutil

def main():
    # 1. Setup Spark Context
    conf = SparkConf().setAppName("PositionalIndex").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # silent 
    # Path Setup
    input_path = "file:///mnt/Dspace/Boody/College/Level 3/1st term/Information Retrieval/project/project/dataset/*.txt"
    # Temporary folder for Spark's messy output
    temp_folder = "spark_output_index"
    # Final clean file you want
    final_output_file = "output.txt"

   # 2. READ & DEBUG
    try:
        raw_data = sc.wholeTextFiles(input_path)
        
        # Check if empty first
        if raw_data.isEmpty():
            print(f"Error: No .txt files found at: {input_path}")
            return

    except Exception as e:
        print(f"Error: {e}")


    # Processing Logic
    def process_file(record):
        file_path, content = record
        doc_name = file_path.split("/")[-1]
        words = content.lower().split()
        output = []
        for index, word in enumerate(words):
            output.append( ((word, doc_name), index + 1) )
        return output

    # 1. Tokenize
    initial_map = raw_data.flatMap(process_file)

    # 2. Sort and Merge
    doc_term_positions = initial_map.groupByKey().mapValues(list)
    term_doc_map = doc_term_positions.map(lambda x: (x[0][0], (x[0][1], x[1])))
    final_index = term_doc_map.groupByKey().mapValues(list)

    # 3. Formatting
    def format_output(record):
        term, doc_list = record
        doc_strings = []
        for doc_name, positions in sorted(doc_list):
            sorted_positions = sorted(positions)
            pos_str = ", ".join(map(str, sorted_positions))
            doc_strings.append(f"{doc_name}: {pos_str}")
        full_doc_str = " ; ".join(doc_strings)
        return f"<{term} : {full_doc_str}>"

    # 4. Final Sort
    final_sorted = final_index.sortByKey().map(format_output)

    # 5. Save
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    final_sorted.coalesce(1).saveAsTextFile(temp_folder)
    
    sc.stop()

    # 6. Cleanup
    print("Spark finished. Cleaning up files...")

    spark_file = os.path.join(temp_folder, "part-00000")

    if os.path.exists(spark_file):
        if os.path.exists(final_output_file):
            os.remove(final_output_file)
        
        shutil.move(spark_file, final_output_file)
        shutil.rmtree(temp_folder)
        print(f"SUCCESS: The file is ready at '{final_output_file}'")
    else:
        print("ERROR: Spark did not generate the expected part-00000 file.")

if __name__ == "__main__":
    main()