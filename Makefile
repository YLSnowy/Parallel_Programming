# Makefile for jit_analysis

# Compiler and compiler flags
CXX=clang++
CXXFLAGS=-std=c++14 -O3 -g -lz -march=native

.PHONY: compile run clean

compile_sse:
	$(CXX) $(CXXFLAGS) ./SSE/gauss_sse.cpp -o ./SSE/gauss_sse

compile_avx:
	$(CXX) $(CXXFLAGS) ./AVX/gauss_avx.cpp -o ./AVX/gauss_avx

compile_avx_512:
	$(CXX) $(CXXFLAGS) ./SSE/gauss_avx_512.cpp -o ./AVX_512/gauss_avx_512

run_sse:
	./SSE/gauss_sse $(SIZE)

run_avx:
	./SSE/gauss_avx $(SIZE)

run_avx_512:
	./SSE/gauss_avx_512 $(SIZE)

clean:
	rm -f $(TARGET)
