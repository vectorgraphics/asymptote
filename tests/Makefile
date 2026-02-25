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

cleaner: FORCE clean

cleanest: FORCE cleaner

distclean: FORCE cleaner
maintainer-clean: FORCE cleanest

FORCE:
