usepackage("movie15","3D");
usepackage("hyperref","setpagesize=false");

// Fix missing BBox bug in movie15 version 2008/01/16
texpreamble("\begingroup\makeatletter%
        \ifpdf%
          \let\@MXV@iiidstream\relax
          \xdef\@MXV@apdict{/AP << /N \@MXV@iiidstream>>}%
        \else
          \pdfmark{%
            pdfmark=/OBJ,%
            Raw={%
              /_objdef {apdict}%
              /type/stream%
            }%
          }%
          \pdfmark{%
            pdfmark=/PUT,%
            Raw={%
              {apdict}%
              <</BBox[0 0 1 1]>>%
            }%
          }%
          \xdef\@MXV@apdict{/AP << /N {apdict}>>}%
        \fi%
\endgroup%");

// See http://www.ctan.org/tex-archive/macros/latex/contrib/movie15/README
// for documentation of the options.

// Embed object in pdf file 
string embed(string name, string options="", real width=0, real height=0)
{
  if(options != "") options="["+options+"]{";
  if(width != 0) options += (string) (width/pt)+"pt"; 
  options += "}{";
  if(height != 0) options += (string) (height/pt)+"pt"; 
  return "\includemovie"+options+"}{"+name+"}";
}

string hyperlink(string url, string text)
{
  return "\href{"+url+"}{"+text+"}";
}

string link(string label, string text, string options="")
{
  // Run LaTeX twice to resolve references.
  settings.twice=true;
  if(options != "") options="["+options+"]";
  return "\movieref"+options+"{"+label+"}{"+text+"}";
}
