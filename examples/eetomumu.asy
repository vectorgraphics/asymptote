import feynman;

// scale all other defaults of the feynman module appropriately
fmdefaults();

// define vertex and external points

real L = 50;

pair zl = (-0.75*L,0);
pair zr = (+0.75*L,0);

pair xu = zl + L*dir(+120);
pair xl = zl + L*dir(-120);

pair yu = zr + L*dir(+60);
pair yl = zr + L*dir(-60);


// draw propagators and vertices

drawFermion(xu--zl);
drawFermion(zl--xl);

drawPhoton(zl--zr);

drawFermion(yu--zr);
drawFermion(zr--yl);

drawVertex(zl);
drawVertex(zr);


// draw momentum arrows and momentum labels

drawMomArrow(xl--zl, left);
label("$k'$", midLabelPoint(xl--zl, right), SE);

label("$k$", midLabelPoint(xu--zl, left), NE);

drawMomArrow(zl--zr, left);
label("$q$", midLabelPoint(zl--zr, right), down);

drawMomArrow(zr--yu, right);
label("$p'$", midLabelPoint(zr--yu, left), NW);

label("$p$", midLabelPoint(zr--yl, right), SW);


// draw particle labels

label("$e^-$", xu, left);
label("$e^+$", xl, left);

label("$\mu^+$", yu, right);
label("$\mu^-$", yl, right);


// shipout
shipout();
