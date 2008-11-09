import stats;
import graph; 
import palette; 
import contour; 
 
size(20cm);

scale(false);

pair[] data=new pair[50000];
for(int i=0; i < data.length; ++i)
  data[i]=Gaussrandpair(); 
 
// Histogram limits and number of bins 
pair datamin=(-0.15,-0.15); 
pair datamax=(0.15,0.15); 
int Nx=30; 
int Ny=30; 

int[][] bins=frequency(data,datamin,datamax,Nx,Ny);
 
real[] values=new real[Nx*Ny]; 
pair[] points=new pair[Nx*Ny];
int k=0; 
real dx=(datamax.x-datamin.x)/Nx;
real dy=(datamax.y-datamin.y)/Ny;
for(int i=0; i < Nx; ++i) {
  for(int j=0; j < Ny; ++j) {
    values[k]=bins[i][j]; 
    points[k]=(datamin.x+(i+0.5)*dx,datamin.y+(j+0.5)*dy); 
    ++k; 
  }
} 
 
// Create a color palette 
pen[] InvGrayscale(int NColors=256) {
  real ninv=1.0/(NColors-1.0); 
  return sequence(new pen(int i) {return gray(1-17*i*ninv);},NColors); 
} 
 
// Draw the histogram, with axes 
bounds range=image(points,values,Range(0,40),InvGrayscale()); 
draw(contour(points,values,new real[] {1,2,3,4,8,12,16,20,24,28,32,36,40},
             operator--),blue); 
xaxis("$x$",BottomTop,LeftTicks,above=true); 
yaxis("$y$",LeftRight,RightTicks,above=true); 

