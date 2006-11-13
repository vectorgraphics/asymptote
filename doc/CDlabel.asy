usepackage("babel","german");
usepackage("fontenc","T1");
size(11.7cm,11.7cm);

asy(nativeformat(),"logo");
fill(unitcircle^^(scale(2/11.7)*unitcircle),
evenodd+rgb(124/255,205/255,124/255));

label(scale(1.1)*minipage(
"\centering\scriptsize \textbf{Nonlinear Modelling, Tutorial and Manual}\\
\textsc{G\"unther H. Mehring}\\
(edited by \textsc{Peter Sch\"opf} and \textsc{Jens Schwaiger})\\
with an \textbf{Appendix} written by\\
\textsc{Wolfgang Prager} and \textsc{Jens Schwaiger}",6cm),(0,0.6));
label(scale(1.1)*minipage("\centering\scriptsize Bericht Nr. 349(2005)\\
{\bfseries Grazer Mathematische Berichte}\\
ISSN 1016--7692",4cm),(0.55,0.2));

label(graphic("logo."+nativeformat(),"height=6cm"),(0,-0.5));
clip(unitcircle^^(scale(2/11.7)*unitcircle),evenodd);
