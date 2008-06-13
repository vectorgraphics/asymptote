# Makefile.

CFLAGS = -O3 -g -Wall
CXX = g++

test: PRCbitStream oPRCFile PRCdouble writePRC test.cc
	$(CXX) $(CFLAGS) -o test PRCbitStream.o oPRCFile.o PRCdouble.o writePRC.o test.cc -lz

PRCbitStream: PRCbitStream.cc
	$(CXX) $(CFLAGS) -c PRCbitStream.cc -o PRCbitStream.o

oPRCFile: oPRCFile.cc
	$(CXX) $(CFLAGS) -c oPRCFile.cc -o oPRCFile.o

PRCdouble: PRCdouble.cc
	$(CXX) $(CFLAGS) -c PRCdouble.cc -o PRCdouble.o

writePRC: writePRC.cc PRCbitStream
	$(CXX) $(CFLAGS) -c writePRC.cc -o writePRC.o

clean:
	rm -f *.o test.prc test
