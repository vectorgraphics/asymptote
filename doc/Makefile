MANFILES = asy.1 xasy.1x
ASYFILES = $(wildcard *.asy) latexusage.asy
INCLUDES = version.texi options
ASY = ../asy -dir ../base
DOCFILES = asymptote.pdf intro.pdf CAD.pdf PDFAnim_temp_Guide.pdf

doc:	asymptote.dvi asymptote.info html man

man:	asymptote.pdf CAD.pdf PDFAnim_temp_Guide.pdf asy.1

faq:	
	cd FAQ; $(MAKE) faq

%.png:	%.asy
	$(ASY) -f png $<

%.eps:	%.asy
	$(ASY) $<

%.pdf:	%.asy
	$(ASY) -f pdf $<

latexusage.dvi: latexusage.tex ../examples/slidedemo.asy
	latex latexusage
	$(ASY) latexusage
	latex latexusage

latexusage.eps: latexusage.dvi
	dvips -o latexusage.eps latexusage

latexusage.png: latexusage.eps
	convert -density 144x144 +antialias -geometry 50%x \
		latexusage.eps latexusage.png
latexusage.pdf: latexusage.dvi
	dvipdf latexusage

options: ../settings.cc
	$(ASY) -h 2>&1 | grep -iv Asymptote > options

asy.1: options asy.1.begin asy.1.end
	cat options | grep ^- | sed -e "s/-\(.*\)  \([A-Z].*\)/.TP\n.B -\1\n\2\./" | sed -e "/^.B/ s/-/\\\\-/g" | cat asy.1.begin - asy.1.end > asy.1

asymptote.dvi: asymptote.texi $(INCLUDES) $(ASYFILES:.asy=.eps)
	texi2dvi asymptote.texi

asymptote.info: asymptote.texi $(INCLUDES) $(ASYFILES:.asy=.png)
	makeinfo asymptote

asymptote.pdf: asymptote.texi $(INCLUDES) $(ASYFILES:.asy=.pdf)
	texi2dvi --pdf asymptote.texi

intro.pdf: intro.asy $(ASYFILES)
	$(ASY) -tex pdflatex intro
	$(ASY) -tex pdflatex intro

intro.png:
	-

CAD.pdf: CAD.tex CAD1.eps 
	latex CAD
	latex CAD
	latex CAD
	dvipdf CAD

PDFAnim_temp_Guide.pdf: PDFAnim_temp_Guide.tex
	latex PDFAnim_temp_Guide
	latex PDFAnim_temp_Guide
	dvipdf PDFAnim_temp_Guide

html:	$(INCLUDES) $(ASYFILES:.asy=.eps) $(ASYFILES:.asy=.png)
	mkdir -p asymptote
	cp *.png asymptote
	makeinfo --html asymptote

clean: FORCE
	rm -f *.eps *.png *.pdf latexusage.{asy,dvi,eps,png,log,aux,*.eps}
	rm -f asymptote.{aux,cp,cps,dvi,fn,info,ky,log,pg,pdf,toc,tp,vr} asymptote.info-*
	rm -f CAD.{aux,dvi,log,pdf,toc}
	rm -rf asymptote

install: man faq
	cd FAQ; $(MAKE) install
	${INSTALL} -d -m 755 $(docdir) $(mandir)/man1
	${INSTALL} -p -m 644 $(DOCFILES) $(docdir)
	${INSTALL} -p -m 644 $(MANFILES) $(mandir)/man1

install-all: doc install
	cd FAQ; $(MAKE) install-info
	${INSTALL} -d -m 755 $(infodir)
	${INSTALL} -p -m 644 asymptote.info $(infodir)
	-${INSTALL} -p -m 644 asymptote.info-* $(infodir)
	-if test -z "$(DESTDIR)"; then \
		install-info --infodir=$(infodir) asymptote.info; \
	fi

uninstall: uninstall-all

uninstall-all:
	cd FAQ; $(MAKE) uninstall
	-cd $(mandir)/man1; rm $(MANFILES)
	-rm -f $(addprefix $(docdir)/,$(DOCFILES))
	-install-info --remove --infodir=$(infodir) asymptote.info
	-rm -f $(infodir)/asymptote.info $(infodir)/asymptote.info-*

distclean: FORCE clean
	rm -f version.texi
	cd FAQ; make distclean
FORCE:
