asyfiles := $(wildcard *.asy)

man:	asymptote.info asymptote.ps html

%.png:	%.eps
	convert -density 288x288 -geometry 25%x $< $@

%.eps:	%.asy
	../asy $<

latexusage.eps: latexusage.tex
	latex latexusage
	../asy latexusage
	latex latexusage
	dvips -o latexusage.eps latexusage
	rm -f latexusage.*.eps latexusage.asy

figures: latexusage.eps latexusage.png $(asyfiles:.asy=.eps) \
	$(asyfiles:.asy=.png)

asymptote.info: figures
	makeinfo asymptote

asymptote.ps: asymptote.dvi
	dvips -o asymptote.ps asymptote

html:	figures
	cp *.png asymptote
	makeinfo --html asymptote

clean: FORCE
	rm -f *.eps *.png latexusage.{asy,dvi,log,aux,ps}
	rm -f asymptote.{aux,cp,cps,dvi,fn,info,ky,log,pg,ps,toc,tp,vr}
	rm -rf asymptote

distclean: FORCE clean
	rm -f version.texi

FORCE:
