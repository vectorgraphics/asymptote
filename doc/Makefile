MANFILES = asy.1 xasy.1x
ASYFILES = $(wildcard *.asy) latexusage.asy
FIGURES = version.texi 

doc:	asymptote.dvi asymptote.info html man

man:	asymptote.pdf

%.png:	%.asy
	../asy -f png $<

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

latexusage.png: latexusage.dvi
	dvipng -o latexusage.png latexusage

latexusage.pdf: latexusage.dvi
	dvipdf latexusage

asymptote.dvi: asymptote.texi $(FIGURES) $(ASYFILES:.asy=.eps)
	texi2dvi asymptote.texi

asymptote.info: asymptote.texi $(FIGURES) $(ASYFILES:.asy=.png)
	makeinfo asymptote

asymptote.pdf: asymptote.texi $(FIGURES) $(ASYFILES:.asy=.pdf)
	texi2dvi --pdf asymptote.texi

html:	$(FIGURES) $(ASYFILES:.asy=.eps) $(ASYFILES:.asy=.png)
	mkdir -p asymptote
	cp *.png asymptote
	makeinfo --html asymptote

clean: FORCE
	rm -f *.eps *.png *.pdf latexusage.{asy,dvi,log,aux,ps,*.eps}
	rm -f asymptote.{aux,cp,cps,dvi,fn,info,ky,log,pg,ps,pdf,toc,tp,vr}
	rm -rf asymptote

install-man: man
	${INSTALL} -m 644 asymptote.pdf $(exampledir)
	${INSTALL} -m 644 $(MANFILES) $(mandir)/man1

install: doc install-man
	${INSTALL} -d $(prefix)/info $(exampledir) $(mandir)/man1
	${INSTALL} -m 644 asymptote.info $(prefix)/info
	install-info --infodir=$(prefix)/info asymptote.info 

uninstall:
	-cd $(mandir)/man1; rm $(MANFILES)
	-rm -f $(exampledir)/asymptote.pdf
	-install-info --delete --infodir=$(prefix)/info doc/asymptote.info
	-rm -f $(prefix)/info/asymptote.info

distclean: FORCE clean
	rm -f version.texi

FORCE:
