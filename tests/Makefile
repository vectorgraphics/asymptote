.NOTPARALLEL:

TESTDIRS = string arith frames types imp array pic gs io

EXTRADIRS = gsl output

test: $(TESTDIRS)

all: $(TESTDIRS) $(EXTRADIRS)

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@/*.asy

$(EXTRADIRS)::
	@echo
	../asy -dir ../base $@/*.asy

clean:  FORCE
	rm -f *.eps

distclean: FORCE clean

FORCE:
