import lmfit;
import graph;

size(10cm, 7cm, IgnoreAspect);

real[] date = { 1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880,
1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990 };
real[] population = { 3.929, 5.308, 7.240, 9.638, 12.866, 17.069, 23.192, 31.443,
38.558, 50.156, 62.948, 75.996, 91.972, 105.711, 122.775, 131.669, 150.697,
179.323, 203.185, 226.546, 248.710 };

real t0 = 1776;

real P(real[] params, real t) {
  real P0 = params[0];
  real K = params[1];
  real r = params[2];
  return (K * P0) / (P0 + (K - P0) * exp(-r * (t - t0)));
}

real[] params = { 10, 500, 0.1 };

real res = lmfit.fit(date, population, P, params).norm;

write("P_0 = ", params[0]);
write("K = ", params[1]);
write("r = ", params[2]);
write("error = ", res);

real P(real t) {
  return P(params, t);
}

draw(graph(date, population), blue);
draw(graph(P, t0, 2000), red);
xaxis("Year", BottomTop, LeftTicks);
yaxis("Population in millions", LeftRight, RightTicks);
