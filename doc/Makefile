MANFILES = asy.1 xasy.1x
ASYFILES = $(wildcard *.asy)

man:	asymptote.info asymptote.ps asymptote.pdf html

%.png:	%.eps
	convert +antialias -density 288x288 -geometry 25%x $< $@

%.eps:	%.asy
	../asy $<

latexusage.asy: latexusage.tex
	latex latexusage
	../asy latexusage

latexusage.eps: latexusage.asy
	latex latexusage
	dvips -o latexusage.eps latexusage
	rm -f latexusage.*.eps

figures: latexusage.eps latexusage.png $(ASYFILES:.asy=.eps) \
	$(ASYFILES:.asy=.png) version.texi

asymptote.info: figures
	makeinfo asymptote

asymptote.dvi: figures asymptote.texi version.texi
	texi2dvi asymptote.texi

asymptote.ps: asymptote.dvi
	dvips -o asymptote.ps asymptote

asymptote.pdf: asymptote.ps
	ps2pdf asymptote.ps

html:	figures
	mkdir -p asymptote
	cp *.png asymptote
	makeinfo --html asymptote

clean: FORCE
	rm -f *.eps *.png latexusage.{asy,dvi,log,aux,ps}
	rm -f asymptote.{aux,cp,cps,dvi,fn,info,ky,log,pg,ps,pdf,toc,tp,vr}
	rm -rf asymptote

install: man
	${INSTALL} -d $(prefix)/info $(exampledir) $(mandir)/man1
	${INSTALL} -m 644 asymptote.info $(prefix)/info
	install-info --infodir=$(prefix)/info asymptote.info 
	${INSTALL} -m 644 asymptote.pdf $(exampledir)
	${INSTALL} -m 644 $(MANFILES) $(mandir)/man1

uninstall:
	-cd $(mandir)/man1; rm $(MANFILES)
	-rm -f $(exampledir)/asymptote.pdf
	-install-info --delete --infodir=$(prefix)/info doc/asymptote.info
	-rm -f $(prefix)/info/asymptote.info

distclean: FORCE clean
	rm -f version.texi

FORCE:
