#example makefile for intel mpi and intel mkl
CTFDIR=${ctfdir}
SRC = matrix.cxx ring.cxx
BIN = matrix.exe ring.exe

.PHONY: all

CXX=mpiicc

all: clean $(BIN)

%.exe: %.cxx
		${CXX}  \
			$<  \
			-std=c++11 \
			-I${CTFDIR}/include \
			-lctf \
			-fopenmp \
			-L${CTFDIR}/lib \
			-Wl,-rpath=${MKLROOT}/lib/intel64 \
			-mkl -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 \
			-O3 \
			-o $@


clean:
		rm -f $(BIN)

