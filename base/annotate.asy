void annotate(picture pic=currentpicture, string title, string text,
              pair position)
{
  pic.add(new void(frame f, transform t) {
      position=t*position;
      label(f,"\special{!/pdfmark where
                 {pop} {userdict /pdfmark /cleartomark load put} ifelse
                 [/Rect["+(string) position.x+" 0 0 "+(string) position.y+"]
                 /Subtype /Text
                 /Name /Comment
                 /Title ("+title+")
                 /Contents ("+text+")
                 /ANN pdfmark}");
    },true);
  draw(pic,position,invisible);
}
