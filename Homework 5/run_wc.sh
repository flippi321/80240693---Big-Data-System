#!/bin/bash

OUTPUTPATH="wc_res/"

echo ===== WordCount JAVA VERSION =====

echo ===== Compile =====
javac -classpath `/hadoop/bin/yarn classpath` WordCount.java
jar cf wc.jar WordCount*.class
echo
echo ===== Clear old output files on HDFS =====
/hadoop/bin/hdfs dfs -rm -r $OUTPUTPATH
echo
echo ===== RUN =====
/hadoop/bin/yarn jar wc.jar WordCount /hw5_data/temp.txt $OUTPUTPATH
echo DONE!
echo ===== Get Result file =====
/hadoop/bin/hdfs dfs -get $OUTPUTPATH .
echo
