
ROOT_DIR= $(shell pwd)

TARGETS= bin/preprocess  bin/make_undirect bin/t2b bin/add_weight bin/bfs bin/wcc bin/pagerank bin/spmv bin/mis bin/radii bin/kcores bin/pagerank_delta

CXX?= g++
CXXFLAGS?= -O3 -Wall -std=c++11 -g -fopenmp -I$(ROOT_DIR)
HEADERS= $(shell find . -name '*.hpp')

all: $(TARGETS)

bin/preprocess: tools/preprocess.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/t2b: tools/t2b.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/add_weight: tools/add_weight.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/make_undirect: tools/make_undirect.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/bfs: examples/bfs.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/wcc: examples/wcc.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/pagerank: examples/pagerank.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/spmv: examples/spmv.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/mis: examples/mis.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/radii: examples/radii.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/kcores: mycode/kcores.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/pagerank_delta: mycode/pagerank_delta.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

clean:
	rm -rf $(TARGETS)
