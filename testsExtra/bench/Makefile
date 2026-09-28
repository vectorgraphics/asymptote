# Benchmarking / profiling helpers for asy.
#
# Default target runs callgrind on a benchmark script and produces an
# annotated, human-readable summary of where asy spends its cycles.
#
# Usage:
#   make                    # build asy with debug symbols, profile, and annotate
#   make callgrind          # produce raw callgrind output only
#   make annotate           # produce annotated text from existing callgrind output
#   make clean              # remove generated profile artifacts
#
# Parameters (override on the command line, e.g. `make BENCH=set.asy`):
#   BENCH         Asymptote source file to profile (default: map.asy)
#   BENCH_OPTS    -u settings string passed to asy (default: maxOperations=500000)
#   ASY_ROOT      Path to the asymptote source root (default: ../..)
#   CFLAGS        Compiler flags used when (re)building asy (default: -g -O2)
#   CALLGRIND_EXTRA_OPTS   Extra flags passed to valgrind --tool=callgrind
#   ANNOTATE_OPTS Flags passed to callgrind_annotate
#                 (default: --auto=yes --threshold=99.5)

BENCH               ?= map.asy
BENCH_OPTS          ?= maxOperations=500000
ASY_ROOT            ?= ../..
CFLAGS              ?= -g -O3
CALLGRIND_EXTRA_OPTS ?=
ANNOTATE_OPTS       ?= --auto=yes --threshold=99.5

ASY                 := $(ASY_ROOT)/asy
ASY_BASE            := $(ASY_ROOT)/base

BENCH_STEM          := $(basename $(BENCH))
CALLGRIND_OUT       := callgrind.out.$(BENCH_STEM)
ANNOTATED_OUT       := callgrind.annotated.$(BENCH_STEM).txt

.PHONY: all asy callgrind annotate clean check-tools top-function

all: annotate

# Build asy via the existing autotools build with debug symbols enabled.
# `make` in the project root is a no-op if asy is already up to date.
asy:
	$(MAKE) -C $(ASY_ROOT) CFLAGS="$(CFLAGS)" asy

# Raw callgrind profile. Depends on asy and the benchmark source.
$(CALLGRIND_OUT): asy $(BENCH)
	@command -v valgrind >/dev/null 2>&1 || { \
	  echo "error: valgrind not found in PATH" >&2; exit 1; }
	rm -f $(CALLGRIND_OUT)
	valgrind --tool=callgrind \
	  --callgrind-out-file=$(CALLGRIND_OUT) \
	  $(CALLGRIND_EXTRA_OPTS) \
	  $(ASY) -dir $(ASY_BASE) -u '$(BENCH_OPTS)' $(BENCH)

callgrind: $(CALLGRIND_OUT)

# Human-readable annotated profile.
$(ANNOTATED_OUT): $(CALLGRIND_OUT)
	@command -v callgrind_annotate >/dev/null 2>&1 || { \
	  echo "error: callgrind_annotate not found in PATH" >&2; exit 1; }
	callgrind_annotate $(ANNOTATE_OPTS) $(CALLGRIND_OUT) > $(ANNOTATED_OUT)
	@echo "Annotated profile written to $(ANNOTATED_OUT)"

annotate: $(ANNOTATED_OUT)

clean:
	rm -f callgrind.out.* callgrind.annotated.*.txt
