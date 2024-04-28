unitsize(1cm);
real gamma=0.5772156649;
import graph;
graph.xaxis(-3.5, 3.5);
graph.yaxis(-2, 2);
xtick("$-\pi$", (-pi, 0), dir=down, red);
xtick("$-\gamma$", (-gamma, 0), dir=down, red);
xtick("$\pi$", (pi, 0), dir=down, heavygreen+basealign);
xtick("$\gamma$", (gamma, 0), dir=down, heavygreen+basealign);
