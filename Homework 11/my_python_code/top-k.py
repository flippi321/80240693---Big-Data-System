from __future__ import print_function
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.dstream import DStream

# --- Code Setup ---

# Initialize SparkContext and StreamingContext
sc = SparkContext(appName="Py_HDFSWordCount")
ssc = StreamingContext(sc, 60)

# Create a DStream that listens to the HDFS directory
hdfs_directory = "hdfs://intro00:8020/user/2024403421/stream" 
lines = ssc.textFileStream(hdfs_directory)

# We use these global variables to keep track of our top-k algorithm
word_counts = {}
file_no = 1
last_file = 5
k = 100

# --- My two functions to perform the Top-K ---

# Function to update the word counts with each new RDD
def update_count(new_counts, last_counts):
    # On first batch we initialize an empty dictionary
    if last_counts is None:
        last_counts = {}

    # On all continuing files, update the word counts
    for word, count in new_counts:
        if word in last_counts:
            last_counts[word] += count
        else:
            last_counts[word] = count

    return last_counts

# Function to process each RDD and compute the top-k frequent words
def process_rdd(time, rdd):
    global word_counts, file_no, last_file, k
    if rdd.isEmpty():
        return

    # Compute word counts for the current RDD
    counts = rdd.flatMap(lambda line: line.split(" ")) \
                .map(lambda word: (word, 1)) \
                .reduceByKey(lambda a, b: a + b)

    # Update the global word counts using the updateCounts function
    updated_counts = counts.collect()  # Collect current counts to update
    word_counts = update_count(updated_counts, word_counts)

    # Sort by count and take top k
    top_k = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:k]

    # We structure the output
    output = f"----- File Number {file_no} -----\n"
    output += f" --- Top-{k} words so far ---\n"
    
    for word, count in top_k:
        output += f"{word}: {count}\n"

    # We write the output to a file
    output_filename = f"output_file_{file_no}.txt"
    with open(output_filename, "w") as f:
        f.write(output)
        
    print(output)

    file_no += 1

    # Stop the program after processing the last file
    if file_no > last_file:
        print("\nMaximum number of files processed. Stopping the streaming.")
        ssc.stop(stopSparkContext=True, stopGraceFully=True)

# --- Use My functions ---

# Process each RDD in the DStream and compute top-k
lines.foreachRDD(process_rdd)

# Start streaming and wait for termination
ssc.start()
ssc.awaitTermination()
