import graph;
import plain_filldraw;


//Will add this to the plain_fill.asy file
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


//3 types of piechart and their labels, nolabel, inside the chart, in legend
struct legendpie {};
struct labelpiein {};
struct labelpieout{};
struct labelpienone {};

legendpie legend1;
labelpiein inside;
labelpieout outside;
labelpienone nolabel;


//for showing percentage
struct percent {};
percent percent;
percent nopercent;

void piechart(picture pic=currentpicture, pair center=(0,0), real radius=1, 
  real[] slicesize, pen[] pens=new pen[] {currentpen}, 
  bool outline=true, labelpienone labeltype=nolabel, percent percentlabel=nopercent,
  Label[] labels=array(slicesize.length,Label("")), pen labelpen=currentpen,
  int[] distances=array(slicesize.length,0)
  ) {
    pens.cyclic=true;
    distances.cyclic=true;
    real angle1=0;
    write(center);

    for(int i=0; i<slicesize.length;++i) {
      real angle2=slicesize[i]*360*0.01; 
      real angle3=((angle1+angle2+angle1)*0.5)*(pi/180);

      real y=distances[i]*sin(angle3);
      real x=distances[i]*cos(angle3);
      
      path arc=shift(x,y)*arc(center,radius,angle1,angle1+angle2);

      fill((x,y)+center--arc--cycle,pens[i]);

      if(outline) {draw((x,y)+center--arc--cycle,labelpen);}

      real yr=radius*sin(angle3);
      real xr=radius*cos(angle3);
      if(percentlabel==percent) {
        label((x+xr*0.65,y+yr*0.65)+center,string(slicesize[i])+"\%",labelpen);}
      angle1+=angle2;
  }
}

void piechart(picture pic=currentpicture, pair center=(0,0), real radius=1, 
  real[] slicesize, pen[] pens=new pen[] {currentpen}, 
  bool outline=true, labelpieout labeltype, percent percentlabel=nopercent,
  Label[] labels=array(slicesize.length,Label("")), pen labelpen=currentpen,
  int[] distances=array(slicesize.length,0)
  ) {
    pens.cyclic=true;
    distances.cyclic=true;
    real angle1=0;

    for(int i=0; i<slicesize.length;++i) {
      real angle2=slicesize[i]*360*0.01; 
      real angle3=((angle1+angle2+angle1)*0.5)*(pi/180);

      real y=distances[i]*sin(angle3);
      real x=distances[i]*cos(angle3);
      
      path arc=shift(x,y)*arc(center,radius,angle1,angle1+angle2);

      fill((x,y)+center--arc--cycle,pens[i]);

      if(outline) {draw((x,y)+center--arc--cycle,labelpen);}

      //this finds the slice that cuts a slice in two
      real yr=radius*sin(angle3);
      real xr=radius*cos(angle3);
      real linedir=(angle3<pi/2 || angle3>=((3pi)/2))?radius*0.4:-radius*0.4;
      dot(((x+xr*0.65,y+yr*0.65)+center));
      draw((x+xr*0.65,y+yr*0.65)+center--(x+xr*1.1,y+yr*1.1)+center--(xr*1.1+(linedir),y+yr*1.1)+center,currentpen);
      if(percentlabel==percent) {
        labels[i].s+="("+string(slicesize[i])+"\%)";}
      label((xr*1.1+(linedir),y+yr*1.1)+center,align=((linedir>0)?E:W),labels[i],labelpen+basealign);
      angle1+=angle2;    
  }
}

void piechart(picture pic=currentpicture, pair center=(0,0), real radius=1, 
  real[] slicesize, pen[] pens=new pen[] {currentpen}, 
  bool outline=true, labelpiein labeltype, percent percentlabel=nopercent,
  Label[] labels=array(slicesize.length,Label("")), pen labelpen=currentpen,
  int[] distances=array(slicesize.length,0)
  ) {
    pens.cyclic=true;
    distances.cyclic=true;
    real angle1=0;

    for(int i=0; i<slicesize.length;++i) {
      real angle2=slicesize[i]*360*0.01; 
      real angle3=((angle1+angle2+angle1)*0.5)*(pi/180);
      real y=distances[i]*sin(angle3);
      real x=distances[i]*cos(angle3);
      path arc=shift(x,y)*arc(center,radius,angle1,angle1+angle2);

      fill((x,y)+center--arc--cycle,pens[i]);
      if(outline) {draw((x,y)+center--arc--cycle,labelpen);}
      real yr=radius*sin(angle3);
      real xr=radius*cos(angle3);
      if(percentlabel==percent) {
        labels[i].s+="("+string(slicesize[i])+"\%)";}
      label((x+xr*0.65,y+yr*0.65)+center,labels[i],labelpen);//figure out the 0.1 number and stuff

      angle1+=angle2;
    }
}

void piechart(picture pic=currentpicture, pair center=(0,0), real radius=1, 
  real[] slicesize, pen[] pens=new pen[] {currentpen},
  bool outline=true, legendpie labeltype, percent percentlabel=nopercent,
  Label[] labels=array(slicesize.length,Label("")),//or Label[] labels=array(slicesize.length, Label(""))
  pen labelpen=currentpen, int[] distances=array(slicesize.length,0)
  ) 
  {//Clashing in the bool statements
    pens.cyclic=true;
    distances.cyclic=true;
    real angle1=0;

    for(int i=0; i<slicesize.length;++i) {

      real angle2=slicesize[i]*360*0.01; 
      real angle3=((angle1+angle2+angle1)*0.5)*(pi/180);
      real y=distances[i]*sin(angle3);
      real x=distances[i]*cos(angle3);
      path arc=shift(x,y)*arc(center,radius,angle1,angle1+angle2);
      if(percentlabel==percent) {
        /*real yr=radius*sin(angle3);
        real xr=radius*cos(angle3);
        label((x+xr*0.65,y+yr*0.65)+center,string(slicesize[i])+"\%",labelpen);*/
        labels[i].s+="("+string(slicesize[i])+"\%)";
      }
      draw((x,y)+center--arc--cycle,pens[i],labels[i]);
      fill((x,y)+center--arc--cycle,pens[i]);

      if(outline) {draw((x,y)+center--arc--cycle,labelpen);}
      
      angle1+=angle2;
    }
    
    add(legend(1,linelength=0.3,black),radius*2NE+center);
}

//i think i dont need to do Label 
Label[] operator cast(string[] labels) {
  Label[] Labels;
  for (int i=0;i<labels.length;++i){
    string a=labels[i];
    Labels.push(Label(a));}
  return Labels;
}








/*void piechart(picture pic=currentpicture, pair center=(0,0), real radius=1, 
  real[] slicesize, pen[] pens=new pen[] {currentpen}, bool outline=true,
  labelpieout labeltype, percent percentlabel=nopercent,
  string[] labels=array(slicesize.length,""), pen labelpen=currentpen,
  int[] distances=array(slicesize.length,0)
  ) {
    pens.cyclic=true;
    distances.cyclic=true;
    real angle1=0;

    for(int i=0; i<slicesize.length;++i) {
      real angle2=slicesize[i]*360*0.01; 
      real angle3=((angle1+angle2+angle1)*0.5)*(pi/180);
      real y=distances[i]*sin(angle3);
      real x=distances[i]*cos(angle3);
      path arc=shift(x,y)*arc(center,radius,angle1,angle1+angle2);

      fill((x,y)--arc--cycle,pens[i]);
      if(outline) {draw((x,y)--arc--cycle,labelpen);}
      real y=radius*sin(angle3);
      real x=radius*cos(angle3);
      if(percentlabel==percent) {
            label((x*0.65,y*0.65),string(slicesize[i])+"\%",labelpen);}

      frame f;
      label(f,labels[i]);
      real offset=size(f).y;
      
      label((x+offset*0.5,y+offset*0.5),labels[i],labelpen);//figure out the 1.2 number and stuff

      angle1+=angle2;
    }
}
void piechart(picture pic=currentpicture, pair center=(0,0), real radius=1, 
  real[] slicesize, pen[] pens=new pen[] {currentpen},
  bool outline=true, bool percent=false, bool legend=true, 
  string[] labels=array(slicesize.length,""), pen labelpen=currentpen,
  int[] distances=array(slicesize.length,0)
  ) {//Clashing in the bool statements
    pens.cyclic=true;
    distances.cyclic=true;
    real angle1=0;

    for(int i=0; i<slicesize.length;++i) {

        real angle2=slicesize[i]*360*0.01; 
        real angle3=((angle1+angle2+angle1)*0.5)*(pi/180);
        real y=distances[i]*sin(angle3);
        real x=distances[i]*cos(angle3);
        path arc=shift(x,y)*arc(center,radius,angle1,angle1+angle2);

        if(legend) {draw((x,y)--arc--cycle,pens[i],labels[i]);}

        fill((x,y)--arc--cycle,pens[i]);

        if (outline){draw((x,y)--arc--cycle,labelpen);}
        
        
        


        if(percent) {
            real y=radius*sin(angle3);
            real x=radius*cos(angle3);
            label((x*0.65,y*0.65),string(slicesize[i])+"\%",labelpen);
        }
        angle1+=angle2;
    }
    
    add(legend(1,linelength=0.3,black),radius*1.55NE);
}
*/
/*
void piechart(picture pic=currentpicture, pair center=(0,0), real radius=1, 
  real[] slicesize, pen[] pens=new pen[] {currentpen},
  bool outline=true, bool percent=false, bool legend=true, 
  string[] labels=array(slicesize.length,""), pen labelpen=currentpen,
  int[] distances=array(slicesize.length,0)
  ) {//Clashing in the bool statements
    pens.cyclic=true;
    distances.cyclic=true;
    real angle1=0;

    for(int i=0; i<slicesize.length;++i) {

        real angle2=slicesize[i]*360*0.01; 
        real angle3=((angle1+angle2+angle1)*0.5)*(pi/180);
        real y=distances[i]*sin(angle3);
        real x=distances[i]*cos(angle3);
        path arc=shift(x,y)*arc(center,radius,angle1,angle1+angle2);

        if(legend) {draw((x,y)--arc--cycle,pens[i],labels[i]);}

        fill((x,y)--arc--cycle,pens[i]);

        if (outline){draw((x,y)--arc--cycle,labelpen);}
        
        
        


        if(percent) {
            real y=radius*sin(angle3);
            real x=radius*cos(angle3);
            label((x*0.65,y*0.65),string(slicesize[i])+"\%",labelpen);
        }
        angle1+=angle2;
    }
    
    add(legend(1,linelength=0.3,black),radius*1.55NE);
}*/
//outer operator cast(int[] int){try.number=int;
//return try.number;} 
//add a key for the colors and stuff
//default for the key is box but can be dots i suppose
//first see if this is wanted maybe ig 
//if they want a key or not 

//instead of bool for key, outline, percent











    /*
    picture key;
    pair height=(radius*0.8,radius*0.8);
    for(int i=slicesize.length-1; i>=0;--i) {
      fill(key,box(height,height+(1.5,1.5)),pens[i]);
      label(key,labels[i],height+(2,-0.2),NE,labelpen+basealign);
      height+=(0,1.5+0.5);
    }
    pair boxSize=size(key).y;
    draw(key,box((radius-3.75,radius-3.75),height+(10)),labelpen);
*/