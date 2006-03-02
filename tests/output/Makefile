# Automated testing to see if the output of Asymptote scripts changes when the
# program is modified.

TESTS=$(basename $(wildcard *.asy))

# This command performs the testing on all scripts.
diff: $(TESTS:=.diff)

# This builds the reference copies of the output using a trusted version of asy
ref: $(TESTS:=.ref)

$(TESTS:=.ref) $(TESTS:=.out): %: 
	@echo Generating $@
	@rm -rf $@
	@mkdir $@
	@cd $@; \
	asy -inlinetex ../$(basename $@)

$(TESTS:=.diff): %.diff: %.out
	diff -I CreationDate -u $(@:.diff=.ref) $(@:.diff=.out)

clean:
	rm -rf *.out

# The reference copies should only be built at the start, or when the behaviour
# of Asymptote is intentionally changed, so they are not usually removed by make
# clean.
veryclean: clean
	rm -rf *.ref

# This tells make to build every dependency from scratch, ignoring the dates on
# files.
.PHONY: $(TESTS:=.ref) $(TESTS:=.out) $(TESTS:=.diff) diff ref clean veryclean
