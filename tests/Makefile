TESTDIRS = string arith frames types imp array gsl

test: $(TESTDIRS)

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@/*.asy

clean:  FORCE
	rm -f *.eps

distclean: FORCE clean

FORCE:
