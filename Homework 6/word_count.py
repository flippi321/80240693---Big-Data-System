import re
import sys
from pyspark import SparkConf, SparkContext
import time

if __name__ == '__main__':
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR") # Added to avoid Warnings cluttering the terminal
    lines = sc.textFile(sys.argv[1])
    top_i_words = 10

    first = time.time()

    # We split the lines into words
    words = lines.flatMap(lambda line: re.split(r'[^\w]+', line))

    # We now count every word
    word_counts = words.countByValue()
    top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:top_i_words]

    # Print the top i words
    print(f"Top {top_i_words} words:")
    for word, count in top_words:
        print(f"({repr(word)}, {count})")

    last = time.time()

    print("Total program time: %.2f seconds" % (last - first))
    sc.stop()
