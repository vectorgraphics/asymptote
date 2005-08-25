MANFILES = asy.1 xasy.1x
ASYFILES = $(wildcard *.asy)
FIGURES = latexusage.eps latexusage.png version.texi $(ASYFILES:.asy=.eps) \

man:	asymptote.dvi asymptote.info asymptote.pdf html

%.png:	%.eps
	convert +antialias -density 288x288 -geometry 25%x $< $@

%.eps:	%.asy
	../asy $<

%.pdf:	%.asy
	../asy -f pdf $<

latexusage.asy: latexusage.tex
	latex latexusage
	../asy latexusage

latexusage.dvi: latexusage.asy
	latex latexusage

latexusage.eps: latexusage.dvi
	dvips -o latexusage.eps latexusage

latexusage.pdf: latexusage.dvi
	dvipdf latexusage

asymptote.dvi: asymptote.texi $(FIGURES)
	texi2dvi asymptote.texi

asymptote.info: asymptote.texi $(FIGURES)
	makeinfo asymptote

asymptote.pdf: asymptote.texi $(FIGURES) $(ASYFILES:.asy=.pdf) 
	texi2dvi --pdf asymptote.texi

html:	$(FIGURES) $(ASYFILES:.asy=.png)
	mkdir -p asymptote
	cp *.png asymptote
	makeinfo --html asymptote

clean: FORCE
	rm -f *.eps *.png *.png latexusage.{asy,dvi,log,aux,ps}
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
