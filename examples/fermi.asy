import feynman;

// set default line width to 0.8bp
currentpen = linewidth(0.8);

// scale all other defaults of the feynman module appropriately
fmdefaults();

// disable middle arrows
currentarrow = None;

// define vertex and external points

pair xu = (-40,+45);
pair xl = (-40,-45);
pair yu = (+40,+45);
pair yl = (+40,-45);

pair zu = (  0,+ 5);
pair zl = (  0,- 5);

// define mid points

pair mxu = (xu+zu)/2;
pair mxl = (xl+zl)/2;
pair myu = (yu+zu)/2;
pair myl = (yl+zl)/2;

// draw fermion lines
drawFermion(xu--zu--yu);
drawFermion(xl--zl--yl);

// draw vertices
drawVertexOX(zu);
drawVertexOX(zl);

// draw gluon. Note that the underlying fermion line is blotted out.
drawGluon(arc((0,0),mxu,myl,CW));

// shipout
