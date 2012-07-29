usepackage("hyperref");
texpreamble("\hypersetup{"+settings.hyperrefOptions+"}");

// Embed object to be run in an external window. An image file name can be
// specified; if not given one will be automatically generated.
string embed(string name, string text="", string options="",
             real width=0, real height=0, string image="")
{
  string options; // Ignore passed options.
  if(image == "") {
    image=stripdirectory(stripextension(name))+"."+nativeformat();
    convert(name+"[0]",image,nativeformat());

    if(!settings.keep) {
      exitfcn currentexitfunction=atexit();
      void exitfunction() {
        if(currentexitfunction != null) currentexitfunction();
        delete(image);
      }
      atexit(exitfunction);
    }
  }
  if(width != 0) options += ", width="+(string) (width/pt)+"pt"; 
  if(height != 0) options +=", height="+(string) (height/pt)+"pt"; 
  return "\href{run:"+name+"}{"+graphic(image,options)+"}";
}

string hyperlink(string url, string text)
{
  return "\href{"+url+"}{"+text+"}";
}

string link(string label, string text="Play")
{
  return hyperlink("run:"+label,text);
}

