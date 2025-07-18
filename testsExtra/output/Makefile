# Automated testing to see if the output of Asymptote scripts changes when the
# program is modified.

# How to call asy from the tests/output/name.out directory
ASY=../../../asy

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
	$(ASY) -keep ../$(basename $@) \
	    >$(basename $@).stdout 2>$(basename $@).stderr; \
	ls >$(basename $@).ls; \
	rm -f *.dvi *.pdf *.gif *.jpg *.jpeg *.png

# Ignore lines with timestamps of the form hh:mm, since the time changes between
# runs.  This regex is fairly broad and it may need to be narrowed.
$(TESTS:=.diff): %.diff: %.out
	diff -I "[0-9][0-9]:[0-9][0-9]" -u $(@:.diff=.ref) $(@:.diff=.out)

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
