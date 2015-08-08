// This file generates a dummy genustwo.pdf file if the rasterized
// image in genustwo.asy failed to render.
file fin = input("genustwo.pdf", check=false);
if (error(fin)) {
  defaultpen(fontsize(20bp));
  label("Rasterized image failed to render.");
  shipout(prefix="genustwo");
} else {
  label("Rasterized image rendered successfully. Please ignore this message.");
}
close(fin);

