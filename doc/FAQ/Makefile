# Python interpreter; overridden by the calling Makefile/CMake via $(PYTHON).
PYTHON ?= python3

BFNNCONV = bfnnconv.py

all: faq texi

# Run twice: the first pass writes asy-faq.xrefdb, the second resolves the
# cross references against it (\qref / \index).
faq: $(BFNNCONV) asy-faq.bfnn
	$(PYTHON) bfnnconv.py asy-faq.bfnn
	$(PYTHON) bfnnconv.py asy-faq.bfnn

texi: asy-faq.texi

# The texi backend isn't part of the default ascii/info/html order, so select
# it explicitly with -only.  Run twice for the same cross-reference reason.
asy-faq.texi: $(BFNNCONV) asy-faq.bfnn
	$(PYTHON) bfnnconv.py -only texi asy-faq.bfnn
	$(PYTHON) bfnnconv.py -only texi asy-faq.bfnn

clean: FORCE
	-rm -f *~ core a.out *.lout *.ps *.ascii *.texi *.xrefdb *.post
	-rm -rf *.html

cleaner: FORCE clean
cleanest: FORCE cleaner
	-rm -f *.info

distclean: FORCE cleaner
maintainer-clean: FORCE cleanest

install-all: install

install: faq
	$(MAKE) install-prebuilt
	${INSTALL} -d -m 755 $(docdir) $(docdir)/asy-faq.html
	${INSTALL} -p -m 644 asy-faq.ascii $(docdir)
	${INSTALL} -p -m 644 asy-faq.html/* $(docdir)/asy-faq.html

install-prebuilt:
	${INSTALL} -d -m 755 $(infodir)
	${INSTALL} -p -m 644 asy-faq.info $(infodir)
	-if test -z "$(DESTDIR)"; then \
		install-info --infodir=$(infodir) asy-faq.info; \
	fi

install-info: faq install-prebuilt

uninstall: uninstall-all

uninstall-all:
	-cd $(docdir)/asy-faq.html && rm -rf *.html
	-cd $(docdir) && rmdir asy-faq.html && rm asy-faq.ascii
	-install-info --remove --infodir=$(infodir) asy-faq.info
	-rm -f $(infodir)/asy-faq.info

FORCE:
