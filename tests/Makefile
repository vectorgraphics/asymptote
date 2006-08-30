TESTDIRS = string arith frames types imp array types

test: $(TESTDIRS)

$(TESTDIRS)::
	../asy $@/*.asy

clean:  FORCE
	rm -f *.eps

distclean: FORCE clean

FORCE:
