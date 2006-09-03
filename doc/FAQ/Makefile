BFNNCONV = bfnnconv.pl m-ascii.pl m-html.pl m-info.pl m-lout.pl m-post.pl

FAQ = asy-faq.html asy-faq.ascii

$(FAQ): $(BFNNCONV) asy-faq.bfnn
	mkdir -p asy-faq.html
	perl bfnnconv.pl asy-faq.bfnn
	perl bfnnconv.pl asy-faq.bfnn

faq:	$(FAQ)

clean: FORCE
	rm -f *~ core a.out *.lout *.ps *.info *.ascii *.xrefdb *.post
	rm -rf *.html

install-all: install install-info

install: faq
	${INSTALL} -d -m 755 $(docdir) $(docdir)/asy-faq.html
	${INSTALL} -p -m 644 asy-faq.ascii $(docdir)
	${INSTALL} -p -m 644 asy-faq.html/* $(docdir)/asy-faq.html

install-info: faq
	${INSTALL} -d -m 755 $(infodir)
	${INSTALL} -p -m 644 asy-faq.info $(infodir)
	-if test -z "$(DESTDIR)"; then \
		install-info --infodir=$(infodir) asy-faq.info; \
	fi

uninstall: uninstall-all

uninstall-all:
	-cd $(docdir)/asy-faq.html; rm -rf *.html
	-cd $(docdir); rmdir asy-faq.html; rm asy-faq.ascii 
	-install-info --delete --infodir=$(infodir) asy-faq.info
	-rm -f $(infodir)/asy-faq.info

distclean: FORCE clean

FORCE:
