.NOTPARALLEL:

# Every subdirectory is a test suite, except the byte-cache Python leaves behind
# once any of the .py scripts here has been imported (by pylint, an editor, or a
# REPL); without the filter `make check` tries to run __pycache__/*.asy and dies.
TESTDIRS = $(filter-out __pycache__/,$(wildcard */))

test: $(TESTDIRS) test_collections_errors test_executable_path test_relocatable

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@*.asy

test_collections_errors:
	@echo
	python3 test_collections_errors.py

# getExecutablePath(), through the sysdir candidate it feeds.  Listed early in
# `test:` above, and .NOTPARALLEL makes prerequisite order the run order, so a
# broken exedir is reported here rather than as a wrong sysdir in every case of
# whatever wider sysdir test follows.
test_executable_path:
	@echo
	python3 test_executable_path.py --asy ../asy --asy-base-dir ../base

# The C="" (CTAN/TeXLive) axis needs a binary with an empty compiled-in sysdir.
# The CMake build has a second executable for it, asy-ctan; here there is only
# ../asy, which *is* that binary exactly when the build was configured
# --enable-texlive-build -- so the whole run doubles as the CTAN run and ../asy
# is handed to the axis directly.  Otherwise the axis reports itself skipped.
CTAN_ASY = $(if $(filter on,$(ASY_TEXLIVE_BUILD)),--asy-ctan ../asy,)

test_relocatable:
	@echo
	python3 test_relocatable.py --asy ../asy --asy-base-dir ../base $(CTAN_ASY)

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
