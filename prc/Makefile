# Makefile.

CFLAGS = -O3 -g -Wall
CXX = g++

MAKEDEPEND = $(CFLAGS) -O0 -M -DDEPEND

FILES = PRCbitStream oPRCFile PRCdouble writePRC

all: $(FILES:=.o)

#test: $(FILES:=.o) test.o
#	$(CXX) $(CFLAGS) -o test $(FILES:=.o) test.o -lz

#test_tess: $(FILES:=.o) test_tess.o
#	$(CXX) $(CFLAGS) -o test_tess $(FILES:=.o) test_tess.o -lz

#test_mesh: $(FILES:=.o) test_mesh.o
#	$(CXX) $(CFLAGS) -o test_mesh $(FILES:=.o) test_mesh.o -lz

.SUFFIXES: .c .cc .o .d
.cc.o:
	$(CXX) $(CFLAGS) $(INCL) -o $@ -c $<
.cc.d:
	@echo Creating $@; \
	rm -f $@; \
	${CXX} $(MAKEDEPEND) $(INCL) $(MDOPTS) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
.c.d:
	@echo Creating $@; \
	rm -f $@; \
	${CC} $(MAKEDEPEND) $(INCL) $(MDOPTS) $< > $@.$$$$ 2>/dev/null && \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

ifeq (,$(findstring clean,${MAKECMDGOALS}))
-include $(FILES:=.d)
endif

clean:
	rm -f *.o *.d test test_tess
