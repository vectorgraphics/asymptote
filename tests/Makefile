.NOTPARALLEL:

TESTDIRS = $(wildcard */)

test: $(TESTDIRS) test_collections_errors test_executable_path

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@*.asy

test_collections_errors:
	@echo
	python3 test_collections_errors.py

# getExecutablePath(), through the one sysdir candidate every build has live --
# so it takes no mode argument: that candidate is outside the IS_RELOCATABLE
# guard.  Listed early in `test:` above, and .NOTPARALLEL makes prerequisite
# order the run order, so a broken exedir is reported here rather than as a
# wrong sysdir in every case of whatever wider sysdir test follows.
test_executable_path:
	@echo
	python3 test_executable_path.py --asy ../asy --asy-base-dir ../base

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
