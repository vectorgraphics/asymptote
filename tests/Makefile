.NOTPARALLEL:

TESTDIRS = $(wildcard */)

test: $(TESTDIRS)

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@*.asy

$(EXTRADIRS)::
	@echo
	../asy -dir ../base $@*.asy

clean:  FORCE
	rm -f *.eps

distclean: FORCE clean

FORCE:
