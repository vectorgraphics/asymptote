import graph;
import three;
import plain_filldraw;


//creates a fill() function that takes filltype as an argument 
//WILL BE MOVED TO  plain_filldraw; (probably or elsewhere)
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
    axialshade(f,g,pena,min(g),penb,max(g));
    //stroke setting for the outline AxialDraw
    });
}

//Create two new structures so then have two functions of barGraph
struct verticalBars {}
struct horinzontalBars {}

verticalBars vertical;
horinzontalBars horizontal;

//barGraph function for vertical bars which is the default when it is called
void barGraph(picture pic=currentpicture, real width, real distance=width,
    real[][] height, filltype[][] filltypes=new filltype[][] {{Draw(currentpen)}},
    string[] labels=array(height.length,""), pen[] labelpens=new pen[] {currentpen}, 
    verticalBars barsDirection=vertical, int clumps=height.length)
{
    for(int i=0;i<filltypes.length;++i) {filltypes[i].cyclic=true;}
    filltypes.cyclic=true;
    labelpens.cyclic=true;

    int numCol=height.length;
    real n=distance==0?width:distance; //to add visually appealing spacing      

    frame f;
    for(int i=0; i < numCol; ++i) {label(f,labels[i],labelpens[i]+basealign);}
    real offset=size(f).y;

    for(int i=0;i<numCol;++i) {
        if(i%clumps==0 & i>1) {n=n+width;}

        real base=0;
        for(int j=0;j<height[i].length;++j) {
            path g=box((n,base),(n+width,base+height[i][j])); 
            fill(pic,g,height[j].length==1?filltypes[j][i]:filltypes[i][j]);
            base += height[i][j];
        }

        label(shift(0,-labelmargin(labelpens[i])-offset)*labels[i],
            (n+(0.5*width),0),N,labelpens[i]+basealign);
        n = width+distance+n;
    }
}


//barGraph for horizontal bars
void barGraph(picture pic=currentpicture, real width, real distance=width,
    real[][] height, filltype[][] filltypes=new filltype[][] {{Draw(currentpen)}},
    string[] labels=array(height.length,""), pen[] labelpens=new pen[] {currentpen}, 
    horinzontalBars barsDirection, int clumps=height.length)
{
    for(int i=0;i<filltypes.length;++i) {filltypes[i].cyclic=true;}
    filltypes.cyclic=true;
    labelpens.cyclic=true;

    int numCol=height.length;
    real n=distance<1?width:distance;        

    frame f;//do i keep this or make it better
    for(int i=0; i < numCol; ++i) label(f,labels[i],labelpens[i]+basealign);
    real offset=size(f).y;
    
    for(int i=0;i<numCol;++i) {
        if (i%clumps==0 & i>1){n=n+width;}

        real base=0;
        for(int j=0;j < height[i].length; ++j) {
            path g=box((base,n),(base+height[i][j],n+width)); 
            fill(pic,g,height[j].length==1?filltypes[j][i]:filltypes[i][j]);
            base+=height[i][j];
    }   
    label(shift(-(offset-3labelmargin(labelpens[i])),0)*labels[i],
        (0,n+(0.5*width)),W,labelpens[i]+basealign);  
    n = width+distance+n;
    }
}


//3D bargraph 
//look at examples of 3d graphs adn stuff 


/*
-sneed to add the basealgin for 3D labels 
-fix the formatting and stuff 
*/


void barGraph3(picture pic=currentpicture, real width=1, real distance=width*2,
    real[][] height, pen[][] pens=new pen[][] {{gray}}, 
    string[] xlabels=array(height.length,""), string[] ylabels=array(height.length,""),
    pen labelpens=currentpen)
{//theres a clash with pen[] (should j be pen) and pen[][] idk if we should complexify it too much 
    for(int i=0;i<pens.length;++i) {pens[i].cyclic=true;}
    pens.cyclic=true;
    
    //to nicely distribute the pens nicely so that {{p,p,p,p,p}} => {{p},{p},{p},{p}} etc
    bool[] penbool=pens[0]==pens[1];
    penbool.cyclic=true;
    pen[] penss=pens[0];
    pen[][] pens;
    if(penbool[height.length-1]) {
        int n=0;
        for(int i; i<height.length; ++i) {
            pen[] pens_;
            for(int j=0; j<height[0].length; ++j) {
                pens_.push(penss[n]); 
                n+=1;
            }
            pens.push(pens_);
        }
    }

    frame f;
    ylabels.cyclic=true;
    for(int i=0; i < height.length; ++i) {label(f,ylabels[i],labelpens+basealign);}
    real offsety=size(f).y;
    frame f;
    xlabels.cyclic=true;
    for(int i=0; i < height.length; ++i) {label(f,xlabels[i],labelpens+basealign);}
    real offsetx=size(f).y;


    real n=0;
    for(int i=0;i<height.length;++i) {
        real m=0;
        for(int j=0;j<height[0].length;++j) {
            surface g=shift(n,m,0)*scale(width,width,height[i][j])*unitcube;
            write(i);
            write(j);
            write(pens[i][j]);
            write("--------------");
            draw(pic,g,meshpen=black+thick(),pens[i][j]);
            label(shift(0,0,-labelmargin(labelpens)-offsety)*ylabels[j],
                (0,width*0.75+m,0),N,labelpens);//figure this out lmaoooO
            m+=distance;
        }

        label(shift(0,-offsetx*0.5-labelmargin(labelpens),0)*xlabels[i],
            (width*.5+n,0,0),labelpens);//IDK shift(0,-offsetx*0.5-labelmargin(labelpens),0)*
        
        //*rotate(-90,(width*0.5+n,0.5,0),(width*0.5+n,0.5,1))
        //label(shift(0,-labelmargin(labelpens)-offsetx,0)*(rotate(-90,(0,-1))*xlabels[i]),((width*0.5)+n,0,0),N,labelpens);//+basealign);
        //still not quite aligned nicely need to like align everything with the x-axis 
        //i need to put rotate first but from there i need to find the new margin i think 
        n+=distance;
        //label(shift(0,-labelmargin(labelpens[i])-offset)*labels[i],(n+(0.5*width),0),N,labelpens[i]+basealign); 
    }   
    
}//maybe make this its own seperate file or something for better height and stuff







//casts

//for the colors and outlines of the 2D bars
filltype[] operator cast(filltype filltype) {return new filltype[] {filltype};}
filltype[][] operator cast(filltype[] filltype) {return new filltype[][] {filltype};}

//
pen[] operator cast(pen pen) {return new pen[] {pen};}
pen[][] operator cast(pen[] pen) {return new pen[][] {pen};}

//for the heights of the bars for 2D and 3D
real[][] operator cast(real[] array) {
    real[][] reals;
    for (int i=0;i<array.length;++i){
        real a=array[i];
        real[] b={a};
        reals.push(b);
    }
    return reals;}








/*
void barGraph(picture pic=currentpicture, real width, real distance=width,//make distance a pen sometimes or nah
            real[] height, filltype[] filltypes=new filltype[] {Draw(currentpen)},
            string[] labels, pen[] labelpens=new pen[] {currentpen}, 
            verticalBars barsDirection=vertical, 

            int layers=0,
            real[] height1=new real[] {0}, filltype[] filltypes1=new filltype[] {Draw(currentpen)},
            real[] height2=new real[] {0}, filltype[] filltypes2=new filltype[] {Draw(currentpen)},
            
            int clumps=height.length)//maybe an array of arrays and depending how many 
{
    filltypes.cyclic=true;
    filltypes1.cyclic=true;
    filltypes2.cyclic=true;
    labelpens.cyclic=true;

    int numCol=height.length;
    frame f;

    for(int i=0; i < numCol; ++i)
        {label(f,labels[i],(0.0),labelpens[i]+basealign);}
    real offset=size(f).y;
    for (int j=0; j < layers+1; ++j){
        height= 0==j?height:height1;
        if (j==1){height=height1;filltypes=filltypes1;}
        if (j==2){height=height2;filltypes=filltypes2;}
        real n=distance<1?width:distance;        
        real w=width;
        for(int i=0; i < numCol; ++i) {
            if (i%clumps==0 & i>1){n=n+width;}
            path g=box((n,0),(n+w,height[i])); 
            label(shift(0,-labelmargin(labelpens[i])-offset)*labels[i],(n+(0.5*w),0),N,labelpens[i]+basealign);
            fill(pic,g,filltypes[i]);
            n = w+distance+n;}
    }   
}*/
/*
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


struct barDirection{int direction;}//int or bool int for more options like the ticks

barDirection vertical;
vertical.direction=1; 
barDirection horizontal;
horizontal.direction=0;
*/

//label(offsetx<70?shift(0,-labelmargin(labelpens[i])-offset)*labels[i]:rotate(-65)*labels[i],(distance+(0.5*w),0),offsetx<70?N:S,labelpens[i]+basealign);


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

    int numCol=height.length;
    real n=distance;
    
    frame f;
    for(int i=0; i < numCol; ++i)
        label(f,labels[i],(0.0),labelpens[i]+basealign);
    real offset=size(f).y;
    real offsetx=size(f).x;
    
    write(offsetx*0.03);
    write(offsetx<70);//how to find width of box based on ??

    for(int i=0; i < numCol; ++i) {
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
