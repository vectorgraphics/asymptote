if(latex() && !settings.inlineimage) {
  usepackage("hyperref");
  texpreamble("\hypersetup{"+settings.hyperrefOptions+"}");
  usepackage("media9","bigfiles");
  texpreamble("\makeatletter%
\newif\ifnoplaybutton
\@ifpackagelater{media9}{2013/11/15}{%
\noplaybuttontrue}{}%
\makeatother%");
}

// For documentation of the options see
// http://www.ctan.org/tex-archive/macros/latex/contrib/media9/doc/media9.pdf

// Embed PRC or SWF content in pdf file 
string embedplayer(string name, string text="", string options="",
                   real width=0, real height=0)
{
  if(width != 0) options += ",width="+(string) (width/pt)+"pt"; 
  if(height != 0) options += ",height="+(string) (height/pt)+"pt"; 
  return "%
\ifnoplaybutton%
\includemedia[noplaybutton,"+options+"]{"+text+"}{"+name+"}%
\else%
\includemedia["+options+"]{"+text+"}{"+name+"}%
\fi";
}

// Embed media in pdf file 
string embed(string name, string text="", string options="",
             real width=0, real height=0)
{
  return embedplayer("VPlayer.swf",text,"label="+name+
                     ",activate=pageopen,addresource="+name+
                      ",flashvars={source="+name+"&scaleMode=letterbox},"+
                     options,width,height);
}

string link(string label, string text="Play")
{
  return "\PushButton[
  onclick={
    annotRM['"+label+"'].activated=true;
    annotRM['"+label+"'].callAS('playPause');
  }]{\fbox{"+text+"}}";
}
