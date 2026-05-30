.NOTPARALLEL:

CXX ?= g++
CXX_STANDARD ?= c++17
ASYBIND_INCLUDE ?= -I../asybind/include

TESTDIRS = $(filter-out cpp-plugins/,$(wildcard */))
CPP_PLUGIN_DIRS = $(wildcard cpp-plugins/*/)
CPP_PLUGIN_CC_FILES = $(wildcard cpp-plugins/*/*.cc)
CPP_PLUGIN_SO_FILES = $(foreach f,$(CPP_PLUGIN_CC_FILES),$(dir $(f))lib$(notdir $(basename $(f))).so)

# Generate a build rule for each plugin .so from its .cc
define cpp_plugin_rule
$(dir $(1))lib$(notdir $(basename $(1))).so: $(1)
	$(CXX) -std=$(CXX_STANDARD) -fPIC -shared $(ASYBIND_INCLUDE) -o $$@ $$<
endef
$(foreach f,$(CPP_PLUGIN_CC_FILES),$(eval $(call cpp_plugin_rule,$(f))))

test: $(TESTDIRS) cpp-plugins test_collections_errors

$(TESTDIRS)::
	@echo
	../asy -dir ../base $@*.asy

cpp-plugins:: $(CPP_PLUGIN_SO_FILES)
	@for d in $(CPP_PLUGIN_DIRS); do \
		echo; \
		(cd $$d && ../../../asy -dir ../../../base *.asy) || exit 1; \
	done

test_collections_errors:
	@echo
	python3 test_collections_errors.py

$(EXTRADIRS)::
	@echo
	../asy -dir ../base $@*.asy

clean:  FORCE
	rm -f *.eps $(CPP_PLUGIN_SO_FILES)

cleaner: FORCE clean

cleanest: FORCE cleaner

distclean: FORCE cleaner
maintainer-clean: FORCE cleanest

FORCE:
