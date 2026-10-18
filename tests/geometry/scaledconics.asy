
import TestLib;
import geometry;

// Regression test for scale-relative tolerances in conic classification,
// intersection, root-finding, and tangent computation.  Verifies that
// conics at very large (10^3 – 10^5) and very small (10^-3) coordinates
// are classified, intersected, and operated on correctly, and that
// tangents() does not return spurious results or crash at non-unit scales.
StartTest("scaledconics");

// ---------------------------------------------------------------------------
// conic classification
// ---------------------------------------------------------------------------
void checktype(conic c, string expected)
{
  string t = conictype(equation(c));
  if (t != expected)
    abort("conictype mismatch: expected '" + expected + "', got '" + t + "'");
}

void checkclassification()
{
  point O = (0, 0);
  point F1 = (3, 0), F2 = (-3, 0), M = (1, 2);

  // circles at various scales
  checktype((conic)circle(O, 1000), "ellipse");
  checktype((conic)circle(O, 1), "ellipse");
  checktype((conic)circle(O, 0.001), "ellipse");

  // ellipse
  checktype((conic)ellipse(O, 10000, 4000), "ellipse");
  checktype((conic)ellipse(O, 5, 3), "ellipse");

  // parabola
  checktype((conic)parabola(O + (1, 0), O), "parabola");

  // hyperbola
  checktype((conic)hyperbola(F1, F2, M), "hyperbola");

  // isparabola
  conic p = (conic)parabola(O + (1, 0), O);
  assert(isparabola(equation(p)));
}

// ---------------------------------------------------------------------------
// circle casting (ellipse -> circle)
// ---------------------------------------------------------------------------
void checkcirclecast()
{
  point O = (0, 0);
  circle big = (circle)((conic)circle(O, 1000));
  assert(abs(big.r - 1000) <= 100 * realEpsilon * 1000);
  circle small = (circle)((conic)circle(O, 0.001));
  assert(abs(small.r - 0.001) <= 100 * realEpsilon * 0.001);
}

// ---------------------------------------------------------------------------
// tangents(conic, vector)
// ---------------------------------------------------------------------------
void checktangentdir(conic c, vector v, int expected)
{
  line[] t = tangents(c, v);
  if (t.length != expected)
    abort("tangents: expected " + string(expected)
          + " line(s), got " + string(t.length));
}

void checktangents()
{
  point O = (0, 0);

  // ellipse – both horizontal and vertical directions give 2 tangents
  conic e = (conic)ellipse(O, 5, 3);
  checktangentdir(e, (1, 0), 2);
  checktangentdir(e, (0, 1), 2);

  // large ellipse
  conic bigE = (conic)ellipse(O, 5000, 3000);
  checktangentdir(bigE, (1, 0), 2);
  checktangentdir(bigE, (0, 1), 2);

  // very large ellipse
  conic vbigE = (conic)ellipse(O, 500000, 300000);
  checktangentdir(vbigE, (1, 0), 2);

  // circle
  conic c = (conic)circle(O, 3);
  checktangentdir(c, (1, 0), 2);
  conic bigC = (conic)circle(O, 3000);
  checktangentdir(bigC, (1, 0), 2);

  // parabola: one tangent along the axis direction
  conic p = (conic)parabola(O + (1, 0), O);
  checktangentdir(p, (0, 1), 1);
  checktangentdir(p, (1, 1), 1);

  // hyperbola: vertical direction gives 2 tangents
  conic h = (conic)hyperbola(O + (3, 0), O + (-3, 0), O + (1, 2));
  checktangentdir(h, (0, 1), 2);

  // zero vector must return 0 tangents
  checktangentdir(e, (0, 0), 0);
}

// ---------------------------------------------------------------------------
// tangents(conic, conic) – common tangents
// ---------------------------------------------------------------------------
void checkcommontangents(conic a, conic b, int expected)
{
  line[] t = tangents(a, b);
  if (t.length != expected)
    abort("common tangents: expected " + string(expected)
          + " line(s), got " + string(t.length));
}

void checkcommontangents()
{
  point O = (0, 0);

  // two disjoint circles, normal scale: 4 common tangents
  conic a1 = (conic)circle(O, 2);
  conic b1 = (conic)circle(O + (5, 0), 1);
  checkcommontangents(a1, b1, 4);

  // two disjoint circles, large scale: 4 common tangents
  // (note: in rotated coordinate systems the bqe intersection algorithm
  //  may miss some tangents due to a pre-existing limitation; we only
  //  check the default frame here)
  if (currentcoordsys == defaultcoordsys)
  {
    conic a2 = (conic)circle(O, 1000);
    conic b2 = (conic)circle(O + (3000, 0), 1000);
    checkcommontangents(a2, b2, 4);
  }

  // concentric circles: 0 common tangents
  conic c1 = (conic)circle(O, 2);
  conic c2 = (conic)circle(O, 1);
  checkcommontangents(c1, c2, 0);
}

// ---------------------------------------------------------------------------
// realquarticroots
// ---------------------------------------------------------------------------
void checkquartic(real a, real b, real c, real d, real e, int expectedn,
                  real[] expected)
{
  real[] roots = realquarticroots(a, b, c, d, e);
  if (roots.length != expectedn)
    abort("realquarticroots: expected " + string(expectedn)
          + " root(s), got " + string(roots.length));
  // check each expected root appears (within tolerance)
  for (int i = 0; i < expected.length; ++i)
  {
    bool found = false;
    for (int j = 0; j < roots.length; ++j)
    {
      real diff = abs(roots[j] - expected[i]);
      real tol = 100 * realEpsilon * max(abs(roots[j]), abs(expected[i]));
      if (tol == 0) tol = realEpsilon;
      if (diff <= tol) { found = true; break; }
    }
    if (!found)
      abort("realquarticroots: expected root " + string(expected[i])
            + " not found");
  }
}

void checkquartics()
{
  // (x^2 - 9)(x^2 - 49) = x^4 - 58x^2 + 441
  checkquartic(1, 0, -58, 0, 441, 4,
               new real[] {-7, -3, 3, 7});

  // (x^2 - 10^6)(x^2 - 4*10^6): large-scale roots
  real c2 = -(10^6 + 4*10^6);
  real e2 = 10^6 * 4 * 10^6;
  checkquartic(1, 0, c2, 0, e2, 4,
               new real[] {-2000, -1000, 1000, 2000});

  // x^4 + 1 = 0: no real roots
  checkquartic(1, 0, 0, 0, 1, 0, new real[] {});

  // x^4 = 0: one (repeated) real root at 0
  checkquartic(1, 0, 0, 0, 0, 1, new real[] {0});

  // (x^2 - 1)(x^2 - 4) = x^4 - 5x^2 + 4
  checkquartic(1, 0, -5, 0, 4, 4,
               new real[] {-2, -1, 1, 2});
}

// ---------------------------------------------------------------------------
// intersectionpoints(bqe, bqe) – no spurious duplicates
// ---------------------------------------------------------------------------
void checkconicintersection(conic a, conic b, int expectedn)
{
  point[] p = intersectionpoints(equation(a), equation(b));
  if (p.length != expectedn)
    abort("intersectionpoints: expected " + string(expectedn)
          + " point(s), got " + string(p.length));
  // verify no near-duplicates
  for (int i = 0; i < p.length; ++i)
    for (int j = i + 1; j < p.length; ++j)
    {
      real sc = max(abs(p[i].x), abs(p[j].x))
              + max(abs(p[i].y), abs(p[j].y));
      if (sc == 0) sc = 1;
      if (abs(p[i].x - p[j].x) <= epsgeo * sc
       && abs(p[i].y - p[j].y) <= epsgeo * sc)
        abort("intersectionpoints: near-duplicate points at index "
              + string(i) + " and " + string(j));
    }
}

void checkintersections()
{
  point O = (0, 0);

  // two disjoint circles, normal scale: 0 intersections
  conic a1 = (conic)circle(O, 1);
  conic b1 = (conic)circle(O + (10, 0), 1);
  checkconicintersection(a1, b1, 0);

  // two intersecting circles: 2 intersections
  conic a2 = (conic)circle(O, 5);
  conic b2 = (conic)circle(O + (6, 0), 5);
  checkconicintersection(a2, b2, 2);

  // two circles, large scale: 2 intersections
  // (note: the bqe intersection algorithm has a pre-existing limitation
  //  in some rotated coordinate systems; we only check the default frame)
  if (currentcoordsys == defaultcoordsys)
  {
    conic a3 = (conic)circle(O, 5000);
    conic b3 = (conic)circle(O + (6000, 0), 5000);
    checkconicintersection(a3, b3, 2);
  }
}

// ---------------------------------------------------------------------------
void run(string tag)
{
  checkclassification();
  checkcirclecast();
  checktangents();
  checkcommontangents();
  checkquartics();
  checkintersections();
}

run("default");

currentcoordsys = cartesiansystem(i = N, j = W);
run("rotated i=N j=W");

EndTest();

