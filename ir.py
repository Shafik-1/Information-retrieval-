from pyspark import SparkContext, SparkConf
import os
import shutil

def main():
    # 1. Setup Spark Context
    conf = SparkConf().setAppName("PositionalIndex").setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # silent 
    # --- PATH SETUP ---
    # Make sure this points to your dataset folder!
    # Based on your previous screenshots, this seems to be your path:
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
            print(f"❌ ERROR: No .txt files found at: {input_path}")
            return

        # --- THE MAGIC DEBUG PART ---
        # collect() brings all data from Spark to Python memory
        all_data = raw_data.collect() 
        
        print(f"✅ Found {len(all_data)} files.\n")
        
        for filename, content in all_data:
            short_name = filename.split("/")[-1]
            # Print filename and first 50 characters of content
            print(f"📄 File: {short_name}") 
            print(f"   Content: {content[:50]}...") 
            print("-" * 20)
        # -----------------------------

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

    print("="*30 + "\n")


    # 3. Processing Logic
    def process_file(record):
        file_path, content = record
        doc_name = file_path.split("/")[-1]
        words = content.lower().split()
        output = []
        for index, word in enumerate(words):
            # index + 1 because positions usually start at 1
            output.append( ((word, doc_name), index + 1) )
        return output

    # 4. Transformations
    initial_map = raw_data.flatMap(process_file)
    doc_term_positions = initial_map.groupByKey().mapValues(list)
    term_doc_map = doc_term_positions.map(lambda x: (x[0][0], (x[0][1], x[1])))
    final_index = term_doc_map.groupByKey().mapValues(list)

    # 5. Formatting
    def format_output(record):
        term, doc_list = record
        doc_strings = []
        for doc_name, positions in sorted(doc_list):
            sorted_positions = sorted(positions)
            pos_str = ", ".join(map(str, sorted_positions))
            doc_strings.append(f"{doc_name}: {pos_str}")
        full_doc_str = " ; ".join(doc_strings)
        return f"<{term} : {full_doc_str}>"

    final_sorted = final_index.sortByKey().map(format_output)

    # 6. Save Logic (Spark part)
    # Remove temp folder if it exists from a failed previous run
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    # Save to the temp folder
    final_sorted.coalesce(1).saveAsTextFile(temp_folder)
    
    # Stop Spark so it releases the files
    sc.stop()

    # 7. AUTOMATIC CLEANUP (The part you asked for)
    print("Spark finished. Cleaning up files...")

    # The file Spark created is typically 'part-00000'
    spark_file = os.path.join(temp_folder, "part-00000")

    if os.path.exists(spark_file):
        # Delete old output.txt if it exists so we can overwrite it
        if os.path.exists(final_output_file):
            os.remove(final_output_file)
        
        # Move the part-00000 file to output.txt
        shutil.move(spark_file, final_output_file)
        
        # Delete the messy folder
        shutil.rmtree(temp_folder)
         sc.stop() call in 
ir.py
 which was causing the AttributeError: 'NoneType' object has no attribute 'setCallSite'.

ir.py
        print(f"SUCCESS: The file is ready at '{final_output_file}'")
    else:
        print("ERROR: Spark did not generate the expected part-00000 file.")

if __name__ == "__main__":
    main()