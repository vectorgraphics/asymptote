.NOTPARALLEL:

# Every subdirectory is a test suite, except the byte-cache Python leaves behind
# once any .py script here has been imported.
TESTDIRS = $(filter-out __pycache__/,$(wildcard */))

test: $(TESTDIRS) test_collections_errors test_executable_path test_relocatable

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@*.asy

test_collections_errors:
	@echo
	python3 test_collections_errors.py

# getExecutablePath(), through the sysdir candidate it feeds.  Listed before
# test_relocatable in `test:` above, and .NOTPARALLEL makes prerequisite order
# the run order, so a broken exedir is reported here rather than as a wrong
# sysdir in every row of the matrix.
test_executable_path:
	@echo
	python3 test_executable_path.py --asy ../asy --asy-base-dir ../base

test_relocatable:
	@echo
	python3 test_relocatable.py --asy ../asy --asy-base-dir ../base

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
