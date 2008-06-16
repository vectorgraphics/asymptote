CFLAGS = -O3 -Wall
CXX = g++

makePRC: PRCbitStream oPRCFile PRCdouble writePRC makePRC.cc
	$(CXX) $(CFLAGS) -o makePRC PRCbitStream.o oPRCFile.o PRCdouble.o writePRC.o makePRC.cc -lz

describePRC: bitData inflation PRCdouble iPRCFile describePRC.cc describeMain.cc
	$(CXX) $(CFLAGS) -o describePRC bitData.o inflation.o PRCdouble.o iPRCFile.o describePRC.cc describeMain.cc -lz

bitSearchUI: bitSearchUI.cc bitData PRCdouble
	$(CXX) $(CFLAGS) -o bitSearchUI bitData.o PRCdouble.o bitSearchUI.cc

bitSearchDouble: bitSearchDouble.cc bitData PRCdouble
	$(CXX) $(CFLAGS) -o bitSearchDouble bitData.o PRCdouble.o bitSearchDouble.cc

extractSections: extractSections.cc iPRCFile inflation bitData PRCdouble
	$(CXX) $(CFLAGS) -o extractSections iPRCFile.o inflation.o bitData.o PRCdouble.o describePRC.cc extractSections.cc -lz

inflateTest: inflation inflationMain.cc
	$(CXX) $(CFLAGS) -o inflateTest inflation.o inflationMain.cc -lz

PRCdouble: ../PRCdouble.cc
	$(CXX) $(CFLAGS) -c ../PRCdouble.cc -o PRCdouble.o

PRCbitStream: ../PRCbitStream.cc
	$(CXX) $(CFLAGS) -c ../PRCbitStream.cc -o PRCbitStream.o

oPRCFile: ../oPRCFile.cc
	$(CXX) $(CFLAGS) -c ../oPRCFile.cc -o oPRCFile.o

writePRC: ../writePRC.cc PRCbitStream
	$(CXX) $(CFLAGS) -c ../writePRC.cc -o writePRC.o

bitData: bitData.cc
	$(CXX) $(CFLAGS) -c bitData.cc -o bitData.o

inflation: inflation.cc
	$(CXX) $(CFLAGS) -c inflation.cc -o inflation.o

iPRCFile: iPRCFile.cc
	$(CXX) $(CFLAGS) -c iPRCFile.cc -o iPRCFile.o

all: makePRC describePRC bitSearchUI bitSearchDouble extractSections inflateTest

tools: all

clean:
	rm -f *.o describePRC bitSearchUI bitSearchDouble extractSections inflateTest
