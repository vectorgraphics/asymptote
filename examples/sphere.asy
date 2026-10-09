import three;

size(200);
currentprojection=orthographic(5,4,3);

if(settings.ibl)
  draw(unitsphere,material(white,shininess=1,metallic=1));
else
  draw(unitsphere,green,render(compression=Zero,merge=true));
