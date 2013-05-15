BFNNCONV = bfnnconv.pl m-ascii.pl m-html.pl m-info.pl m-lout.pl m-post.pl

all: faq

faq: $(BFNNCONV) asy-faq.bfnn
	mkdir -p asy-faq.html
	perl bfnnconv.pl asy-faq.bfnn
	perl bfnnconv.pl asy-faq.bfnn

clean: FORCE
	-rm -f *~ core a.out *.lout *.ps *.info *.ascii *.xrefdb *.post
	-rm -rf *.html

install-all: install

install: faq install-prebuilt
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

distclean: FORCE clean

FORCE:
