import graph3;
import palette;

size(200);
currentprojection=perspective(4,2,3);
currentlight=White;

// Unit cylinder bounding box corners: x,y in [-1,1], z in [0,1]
triple[] coords={
  (-1,-1,0), (1,-1,0), (-1,1,0), (1,1,0),
  (-1,-1,1), (1,-1,1), (-1,1,1), (1,1,1)
};
coords = zscale3(0.7) * ((coords - (0,0,0.5)) / sqrt(2)) + (0,0,0.5); // Scale to fit in unit cylinder, then shift back to original z range.

pen[] cols={
  rgb(0.9, 0.1, 0.1),   // (-1,-1,0) SW: fire red      } z=0: warm fire wheel,
  rgb(1.0, 0.45,0.0),   // ( 1,-1,0) SE: orange        }   hues run CW around circle
  rgb(0.85,0.1, 0.55),  // (-1, 1,0) NW: hot pink      }
  rgb(1.0, 0.85,0.0),   // ( 1, 1,0) NE: golden yellow }
  rgb(0.5, 0.0, 0.85),  // (-1,-1,1) SW: violet        } z=1: aurora wheel,
  rgb(0.05,0.2, 0.9),   // ( 1,-1,1) SE: cobalt blue   }   hues run CCW (reversed!)
  rgb(0.1, 0.85,0.3),   // (-1, 1,1) NW: electric green}
  rgb(0.0, 0.8, 0.75),  // ( 1, 1,1) NE: aqua teal     }
};
// z=0 theme: fire (warm); z=1 theme: aurora (cool).
// The angular hue direction reverses top-to-bottom, creating a spiral that
// requires the xy, xz, yz, xyz cross-terms -- not reproducible analytically.

spatialPen cpen=fitColors(coords,cols);

// Build a finely tessellated cylinder as a surface of revolution:
// ncirc patches around the circumference, nvert patches vertically.
int ncirc=64, nvert=32;
path3 gen = operator--(...sequence(new triple(int i) { return (1,0,i/nvert); }, nvert+1));
surface cyl=surface(O, gen, Z, ncirc);

draw(cyl, emissive(black), spatialpen=cpen);
for (int i=0; i < 8; ++i) {
  dot(coords[i], cols[i]);
}
