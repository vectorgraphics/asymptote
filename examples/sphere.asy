import three;

size(200);
currentprojection=orthographic(5,4,3);

draw(unitsphere,green+opacity(0.5),render(compression=Zero,merge=true));
//draw(unitsphere,material(white,shininess=1,metallic=1));
