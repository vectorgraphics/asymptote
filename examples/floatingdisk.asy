import trembling; 
if(settings.outformat == "") 
  settings.outformat="pdf"; 

size(6cm,0); 
 
real R=1/5; 
real h=0.5; 
real d=1/12; 
real l=.7; 
 
pair pA=(-l,0); 
pair pB=(l,0); 

tremble tr=tremble(angle=10,frequency=0.1,random=50,fuzz=1);
path waterline=tr.deform(pA..pB); 

path disk=shift(0,-d)*scale(R)*unitcircle; 
path water=waterline--(l,-h)--(-l,-h)--(-l,0)--cycle; 
path container=(l,1/7)--(l,-h)--(-l,-h)--(-l,1/7); 
 
filldraw(disk,red,linewidth(.3)); 
fill(water,mediumgrey+opacity(0.5)); 
draw(waterline); 
 
draw(container,linewidth(1.5)); 
 
shipout(bbox(2mm));
