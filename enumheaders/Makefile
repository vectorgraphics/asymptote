PYTHON = python3

all: headers asy

HEADERS = v3dheadertypes v3dtypes
headers: $(HEADERS:=.h)

asy: $(HEADERS:=.asy)

%.h: %.csv
	$(PYTHON) ../generate_enums.py --language cxx -o $@ -i $< --name $(notdir $*) --xopt namespace=camp

%.asy: %.csv
	$(PYTHON) ../generate_enums.py --language asy -o $@ -i $< --name $(notdir $*)

clean:
	rm -rf *.h *.asy
