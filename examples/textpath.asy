size(300);

fill(texpath(Label("test",TimesRoman())),pink);
fill(texpath(Label("test",fontcommand('.fam T\n.ps 12')),tex=false),red);

pair z=10S;

fill(texpath(Label("$ \sqrt{x^2} $",z,TimesRoman())),pink);
fill(texpath(Label("$ sqrt {x sup 2} $",z,fontcommand('.fam T\n.ps 12')),
             tex=false),red);
