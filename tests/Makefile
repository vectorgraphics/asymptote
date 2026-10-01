.NOTPARALLEL:

TESTDIRS = $(wildcard */)

test: $(TESTDIRS) test_collections_errors test_render3d

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@*.asy

test_collections_errors:
	@echo
	python3 test_collections_errors.py

test_render3d:
	@echo
	python3 test_render3d.py || test $$? -eq 77

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
