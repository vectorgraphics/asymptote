import graph;
import plain_filldraw;
/*
Ideas:
-horizontal vs vertical bars
-cast for them ^
*/

void fill(picture pic=currentpicture, path[] g, filltype filltype,
          bool copy=true)
{
  if(copy)
    g=copy(g);
  pic.add(new void(frame f, transform t) {
      filltype.fill(f,t*g,nullpen);
    },true);
  pic.addPath(g);
}

frame f;
filltype AxialShade(pen pena, pen penb)
{
  return filltype(new void(frame f, path[] g, pen) {
    axialshade(f,g,pena,min(g),penb,max(g));//stroke setting for the outline AxialDraw
    });
}

filltype RadialShade(pen penc, pen penr)
{
  return filltype(new void(frame f, path[] g, pen) {
    pair c=(min(g)+max(g))/2;
    radialshade(f,g,penc,c,0,penr,c,abs(max(g)-min(g))/2);
    });
}

/*
//this would create two new structures so then have two functions or cast omg i could cast great idea apiphiny moment
struct verticalBars {}
struct horinzontalBars {}

verticalBars vertical;
horinzontalBars horizontal;

//for now can cast it to a bool can create different statemnts later when needed
bool operator cast(verticalBars vertical){return false;}
bool operator cast(horinzontalBars horizontal){return true;}
//think of a return thats numeric or something such that the value can be used to graph the bars maybe
//or to leave more possibilities than just true or false 

struct barDirection{ 
    verticalBars vert;
    horinzontalBars hori;
}

barDirection vertical;
vertical.vert.reflect; //is a pair of what could be used on the reflect function
barDirection horinzontal;
horinzontal.hori;
*/

struct barDirection{int direction;}//int or bool int for more options like the ticks

barDirection vertical;
vertical.direction=1; 
barDirection horizontal;
horizontal.direction=0;


void barGraph(picture pic=currentpicture, real width, real distance=width,//make distance a pen sometimes or nah
            real[] height, filltype[] filltypes=new filltype[] {Draw(currentpen)},
            string[] labels, pen[] labelpens=new pen[] {currentpen}, 
            barDirection barsDirection=vertical)//options rn are vertical or horizontal
{
    filltypes.cyclic=true;
    labelpens.cyclic=true;

    int NumCol=height.length;
    real n=distance<1?width:distance;

    frame f;
    for(int i=0; i < NumCol; ++i)
        label(f,labels[i],(0.0),labelpens[i]+basealign);
    real offset=size(f).y;
    
    for(int i=0; i < NumCol; ++i) {
        path g=box((n,0),(n+width,height[i])); //is it better to create this based on vert or hor etc. or if elif statements so that if something else is added can insert new elif
        if (barsDirection.direction==0){
            g=reflect((0,0),(max(height),max(height)))*g;//idk if this way works the best tho 
            label(labels[i],(0,n+(0.25*width)),NW,
                labelpens[i]+basealign); //add pen labelpen?
        }
        else{label(shift(0,-labelmargin(labelpens[i])-offset)*labels[i],
            (n+(0.5*width),0),N,labelpens[i]+basealign);}
        //label(offsetx<70?shift(0,-labelmargin(labelpens[i])-offset)*labels[i]:rotate(-65)*labels[i],(distance+(0.5*width),0),offsetx<70?N:S,labelpens[i]+basealign);

        fill(pic,g,filltypes[i]);
        n = width+distance+n;
    }
}

filltype[] operator cast(filltype filltype) {
    return new filltype[] {filltype};
    }








//instead of the next few lines, cast vertical bars into one thing and horizontal into another thing 
/*struct barDirection { 
    string direction;
    void operator init(string direction){
        this.direction=direction;
    }
}
barDirection vertical;
vertical.direction="vertical";
barDirection horizontal;
horizontal.direction="horizontal";

void barGraph(picture pic=currentpicture, real width, real distance=width,
            real[] height, string[] labels, pen[] drawpens, pen[] fillpens, pen[] labelpens=drawpens,
            bool direction=vertical//vertical or angle create a structure for more than two choices 
            //and create instances for each type for now we have 2 but more may be needed 
            )
{
    drawpens.cyclic=true;
    fillpens.cyclic=true;
    labelpens.cyclic=true;

    int NumCol=height.length;
    real n=distance;
    
    frame f;
    for(int i=0; i < NumCol; ++i)
        label(f,labels[i],(0.0),labelpens[i]+basealign);
    real offset=size(f).y;
    real offsetx=size(f).x;
    
    write(offsetx*0.03);
    write(offsetx<70);//how to find width of box based on ??

    for(int i=0; i < NumCol; ++i) {
        path g=box((distance,0),(distance+width,height[i]));
        if (!direction){
            g=reflect((0,0),(max(height),max(height)))*g;
            label(labels[i],(0,distance+(0.25*width)),NW,labelpens[i]+basealign); //add pen labelpen?
        }
        else {label(offsetx<70?shift(0,-labelmargin(labelpens[i])-offset)*labels[i]:rotate(-65)*labels[i],(distance+(0.5*width),0),offsetx<70?N:S,labelpens[i]+basealign);
        }//label(pic, rotate(-80)*labels[i],(distance+(0.15*width),0),S, labelpen[i]);
        //seems to be rotating around the start and not center 

        filldraw(pic,g,fillpens[i],drawpens[i]);
        
        distance = width + distance + n;
    }
}

pen[] operator cast(pen p) {return new pen[] {p};};
// create second barGraph that calls the first (so can keep pen and pen[])
void barGraph(picture pic=currentpicture, real width, real distance=width,
            real[] height,string[] labels, pen drawpens=currentpen, pen fillpen=nullpen, pen labelpen=currentpen, bool direction=true){ 
        //one with the default statements of currentpen and nullpen
        barGraph(pic,width,distance,height,labels,new pen[] {drawpens},new pen[] {fillpen},labelpen,direction);
     }

//array of fillpens
void barGraph(picture pic=currentpicture,real width,real distance=width,real[] height,
            string[] labels,pen drawpens=currentpen,pen[] fillpen, pen labelpen=currentpen, bool direction=true){
        barGraph(pic,width,distance,height,labels,new pen[] {drawpens},fillpen,labelpen,direction);
     }

//array of drawpens 
void barGraph(picture pic=currentpicture,real width,real distance=width,real[] height,
            string[] labels,pen[] drawpens,pen fillpen=nullpen, pen labelpen=currentpen, bool direction=true){
        barGraph(pic,width,distance,height,labels,drawpens,new pen[] {fillpen},labelpen,direction);
     }
*/
