TESTDIRS = string arith frames types imp array types

test: $(TESTDIRS)

$(TESTDIRS)::
	../asy -dir ../base $@/*.asy

clean:  FORCE
	rm -f *.eps

distclean: FORCE clean

FORCE:
