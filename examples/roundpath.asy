// example file for 'roundedpath.asy'
// written by stefan knorr

// import needed packages
import roundedpath;

// define open and closed path
path A = (0,0)--(10,10)--(30,10)--(20,0)--(30,-10)--(10,-10);
path B = A--cycle;

draw(shift(-60,0)*A, green);
draw(shift(-30,0)*roundedpath(A,1), red);

// draw open path and some modifications
for (int i = 1; i < 20; ++i)
  draw(roundedpath(A,i/4), rgb(1 - i*0.049, 0, i*0.049) + linewidth(0.5));

draw(shift(-60,-30)*B, green);
draw(shift(-30,-30)*roundedpath(B,1), red);

//draw closed path and some modifications
for (int i = 1; i < 20; ++i)                          // only round edges
  draw(shift(0,-30)*roundedpath(B,i/4), rgb(0.5, i*0.049,0) + linewidth(0.5));

for (int i = 1; i < 20; ++i)                          // round edged and scale 
  draw(shift(0,-60)*roundedpath(B,i/4,1-i/50), rgb(1, 1 - i*0.049,i*0.049) + linewidth(0.5));

for (int i = 1; i < 50; ++i)                          // shift (round edged und scaled shifted version)
  draw(shift(-30,-60)*shift(10,0)*roundedpath(shift(-10,0)*B,i/10,1-i/80), rgb( i*0.024, 1 - i*0.024,0) + linewidth(0.5));

for (int i = 1; i < 20; ++i)                          // shift (round edged und scaled shifted version)
  draw(shift(-60,-60)*shift(10,0)*roundedpath(shift(-10,0)*B,i/4,1-i/50), gray(i/40));
