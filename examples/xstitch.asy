pair c=(0,0.8);

int iters(pair z, int max=160) {
  int n=0;
  while(abs(z) < 2 && n < max) {
    z=z*z+c;
    ++n;
  }
  return n;
}

int[] cutoffs={12,15,20,30,40,60,200};
int key(pair z) {
  int i=iters(z);
  int j=0;
  while(cutoffs[j] < i)
    ++j;
  return j;
}


int width=210;
int height=190;

real zoom=2.5/200;

int[][] values=new int[width][height];
int[] histogram;  for(int v=0; v < 10; ++v) histogram.push(0);
for(int i=0; i < width; ++i) {
  real x=zoom*(i-width/2);
  for(int j=0; j < height; ++j) {
    real y=zoom*(j-height/2);
    int v=key((x,y));
    values[i][j]=v;
    ++histogram[v];
  }
}

// Print out a histogram.
write("histogram: ");
write(histogram);


pen linepen(int i, int max) {
  real w=i == -1 || i == max+1   ? 2.0 : 
    i % 10 == 0 || i == max ? 1.0 : 
    i % 5 == 0              ? 0.8 : 
    0.25;
  return linewidth(w);
}

pen xpen(int i) {
  return linepen(i,width)+(i == width/2 ? red : 
                           i == 75 || i == width-75 ? dashed : 
                           black);
}

pen ypen(int i) {
  return linepen(i,height)+(i == height/2 ? red : 
                            i == 75 || i == height-75 ? dashed : 
                            black);
}

// The length of the side of a cross stitch cell.
real cell=2.3mm;
transform t=scale(cell);


picture tick;
draw(tick,(0,0)--(1,1));

picture ell;
draw(ell,(0,1)--(0,0)--(0.7,0));

picture cross;
draw(cross,(0,0)--(1,1));
draw(cross,(1,0)--(0,1));

picture star;
draw(star,(0.15,0.15)--(0.85,0.85));
draw(star,(0.85,0.15)--(0.15,0.85));
draw(star,(.5,0)--(.5,1));
draw(star,(0,.5)--(1,.5));

picture triangle;
draw(triangle,(0,0)--(2,0)--(1,1.5)--cycle);

picture circle;
fill(circle,shift(1,1)*unitcircle);

picture ocircle;
draw(ocircle,shift(1,1)*unitcircle);

picture spare;
fill(spare,(0,0)--(1,1)--(0,1)--cycle);

picture[] pics={tick,ell,cross,star,triangle,circle};
pen[] colors={black,0.2purple,0.4purple,0.6purple,0.8purple,purple,
              0.8purple+0.2white};

frame[] icons;
icons.push(newframe);
for(picture pic : pics) {
  // Scaling factor, so that we don't need weird line widths.
  real X=1.0;
  frame f=pic.fit(.8X*cell,.8X*cell,Aspect);
  f=scale(1/X)*f;

  // Center the icon in the cell.
  f=shift((cell/2,cell/2)-0.5(max(f)-min(f)))*f;

  icons.push(f);
}

void drawSection(int xmin, int xmax, int ymin, int ymax) {
  static int shipoutNumber=0;

  // Draw directly to a frame for speed reasons.
  frame pic;

  for(int i=xmin; i <= xmax; ++i) {
    draw(pic,t*((i,ymin)--(i,ymax)),xpen(i));
    if(i%10 == 0) {
      label(pic,string(i),t*(i,ymin),align=S);
      label(pic,string(i),t*(i,ymax),align=N);
    }
  }
  for(int j=ymin; j <= ymax; ++j) {
    draw(pic,t*((xmin,j)--(xmax,j)),ypen(j));
    if(j%10 == 0) {
      label(pic,string(j),t*(xmin,j),align=W);
      label(pic,string(j),t*(xmax,j),align=E);
    }
  }

  if(xmin < 0)
    xmin=0;
  if(xmax >= width)
    xmax=width-1;
  if(ymin < 0)
    ymin=0;
  if(ymax >= height)
    ymax=height-1;

  int stitchCount=0;
  path box=scale(cell) *((0,0)--(1,0)--(1,1)--(0,1)--cycle);
  for(int i=xmin; i < xmax; ++i)
    for(int j=ymin; j < ymax; ++j) {
      int v=values[i][j];
      add(pic,icons[v],(i*cell,j*cell));
      //fill(pic,shift(i*cell,j*cell)*box,colors[v]);
      if(v != 0)
        ++stitchCount;
    }

  write("stitch count: ",stitchCount);

  //  shipout("xstitch"+string(shipoutNumber),pic);
  shipout(pic);
  ++shipoutNumber;
}

//drawSection(-1,width+1,-1,height+1);


//drawSection(-1,80,height-80,height+1);
//drawSection(70,150,height-80,height+1);
drawSection(quotient(width,2)-40,quotient(width,2)+40,quotient(height,2)-40,quotient(height,2)+40);
//drawSection(width-150,width-70,-1,80);
//drawSection(width-80,width+1,-1,80);
