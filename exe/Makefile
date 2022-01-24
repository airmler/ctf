CTFDIR=/home/irmler/src/ctf/

SRC = test.cxx
BIN = test.exe

.PHONY: all


all: clean $(BIN)

%.exe: %.cxx
		mpicxx  \
			$<  \
			-std=c++11 \
			-I${CTFDIR}/include \
			-lctf \
			-fopenmp \
			-L${CTFDIR}/lib \
			-L/usr/lib/x86_64-linux-gnu/libopenblas.a \
			-lopenblas \
			-O0 -g \
			-o $@


clean:
		rm -f $(BIN)

