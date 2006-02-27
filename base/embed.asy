texpreamble("\usepackage[3D]{movie15}
\usepackage{hyperref}");

string embed(string name, string options="", real width=0, real height=0)
{
  if(options != "") options="["+options+"]{";
  if(width != 0) options += (string) (width*pt)+"pt"; 
  options += "}{";
  if(height != 0) options += (string) (height*pt)+"pt"; 
  return "\includemovie"+options+"}{"+name+"}";
}

string hyperlink(string label, string text, string options="")
{
  if(options != "") options="["+options+"]";
  return "\movieref"+options+"{"+label+"}{"+text+"}";
}
