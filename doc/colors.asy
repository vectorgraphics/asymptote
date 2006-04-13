int i=0;
int j=0;

void col(pen p, string s, bool fillblack=false) {
 j -= 10;
// real[] a=colors(p);
// for(int i=0; i < a.length; ++i)
//   s += " "+(string) a[i];
 if(fillblack) label(s,(i+10,j),E,p,Fill(black));
 else label(s,(i+10,j),E,p);
 fill(box((i,j-5),(i+10,j+5)),p);
}

col(black,"black");
col(gray,"gray");
col(white,"white",fillblack=true);

col(palered,"palered");
col(salmon,"salmon");
col(mediumred,"mediumred");
col(red,"red");
col(strongred,"strongred");
col(brown,"brown");
col(darkbrown,"darkbrown");
j -= 10;

col(palegreen,"palegreen");
col(lightgreen,"lightgreen");
col(mediumgreen,"mediumgreen");
col(green,"green");
col(stronggreen,"stronggreen");
col(deepgreen,"deepgreen");
col(darkgreen,"darkgreen");
j -= 10;

col(paleblue,"paleblue");
col(lightblue,"lightblue");
col(mediumblue,"mediumblue");
col(blue,"blue");
col(strongblue,"strongblue");
col(deepblue,"deepblue");
col(darkblue,"darkblue");
j -= 10;

i += 150;
j=0;

col(palecyan,"palecyan");
col(lightcyan,"lightcyan");
col(mediumcyan,"mediumcyan");
col(cyan,"cyan");
col(strongcyan,"strongcyan");
col(deepcyan,"deepcyan");
col(darkcyan,"darkcyan");
j -= 10;

col(pink,"pink");
col(lightmagenta,"lightmagenta");
col(mediummagenta,"mediummagenta");
col(magenta,"magenta");
col(strongmagenta,"strongmagenta");
col(deepmagenta,"deepmagenta");
col(darkmagenta,"darkmagenta");
j -= 10;

col(paleyellow,"paleyellow");
col(lightyellow,"lightyellow");
col(mediumyellow,"mediumyellow");
col(yellow,"yellow");
col(lightolive,"lightolive");
col(olive,"olive");
col(darkolive,"darkolive");
j -= 10;

i += 150;
j=0;

col(orange,"orange");
col(fuchsia,"fuchsia");
j -= 10;
col(chartreuse,"chartreuse");
col(springgreen,"springgreen");
j -= 10;
col(purple,"purple");
col(royalblue,"royalblue");
j -= 10;

col(Cyan,"Cyan");
col(Magenta,"Magenta");
col(Yellow,"Yellow");
col(Black,"Black");

col(cmyk+red,"cmyk+red");
col(cmyk+blue,"cmyk+blue");
col(cmyk+green,"cmyk+green");
