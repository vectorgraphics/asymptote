BFNNCONV = bfnnconv.pl m-ascii.pl m-html.pl m-info.pl m-lout.pl m-post.pl

faq:	asy-faq.html asy-faq.ascii

asy-faq.html: $(BFNNCONV) asy-faq.bfnn
	perl bfnnconv.pl asy-faq.bfnn
	perl bfnnconv.pl asy-faq.bfnn

asy-faq.ascii: $(BFNNCONV) asy-faq.bfnn
	perl bfnnconv.pl asy-faq.bfnn
	perl bfnnconv.pl asy-faq.bfnn

clean: FORCE
	rm -f *~ core a.out *.lout *.ps *.info *.ascii *.xrefdb *.post
	rm -rf *.html

install: faq
	${INSTALL} -d -m 755 $(docdir) $(docdir)/asy-faq.html
	${INSTALL} -p -m 644 asy-faq.ascii $(docdir)
	${INSTALL} -p -m 644 asy-faq.html/* $(docdir)/asy-faq.html

uninstall:
	-cd $(docdir)/asy-faq.html; rm -rf *.html
	-cd $(docdir); rmdir asy-faq.html; rm asy-faq.ascii 

distclean: FORCE clean

FORCE:
