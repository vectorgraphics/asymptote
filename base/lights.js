for(var i=scene.lights.count-1; i >= 0; i--)
  scene.lights.removeByIndex(i);

l1=scene.createLight();
l1.color.set(1,1,1);
l1.brightness=0.6;
l1.direction.set(-0.235702260395516,0.235702260395516,-0.942809041582063)

l2=scene.createLight();
l2.color.set(1,1,1);
l2.brightness=0.6;
l2.direction.set(-0.235702260395516,0.235702260395516,0.942809041582063)

l3=scene.createLight();
l3.color.set(1,1,1);
l3.brightness=0.6;
l3.direction.set(0.235702260395516,-0.235702260395516,0.942809041582063)

l4=scene.createLight();
l4.color.set(1,1,1);
l4.brightness=0.6;
l4.direction.set(0.235702260395516,-0.235702260395516,-0.942809041582063)

// Work around apparent bug in Adobe Reader 8.0:
scene.lightScheme=scene.LIGHT_MODE_HEADLAMP;

scene.lightScheme=scene.LIGHT_MODE_FILE;

