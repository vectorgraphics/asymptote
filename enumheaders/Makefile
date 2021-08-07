PYTHON = python3

default: headers

HEADERS = v3dheadertypes v3dtypes
headers: $(HEADERS:=.h)

%.h: %.csv
	$(PYTHON) ../generate_enums.py --language cxx -o $@ -i $< --name $(notdir $*) --xopt namespace=camp

clean:
	rm -rf *.h