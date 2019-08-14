var p = [
    [0, 0, 1.6],
    [1, 1.333333333333333, 0],
    [0, 0.666666666666667, 0],
    [0, 1, 0],
    [0.333333333333333, 0, 0],
    [0.333333333333333, 0.333333333333333, 0],
    [0.333333333333333, 0.666666666666667, 0],
    [0.333333333333333, 1, 0],
    [0.666666666666667, 0, 0],
    [0.666666666666667, 0.333333333333333, 0],
    [0.666666666666667, 0.666666666666667, 0],
    [0.666666666666667, 1, 0],
    [1, 0, 0],
    [1, 0.333333333333333, 0],
    [1, 0.666666666666667, 0],
    [1, 1, 0.1]
  ];

  var materialIndex = 0;

  var objMaterial = new Material(
    baseColor = [1, 1, 0, 1],
    emissive = [0, 0, 0, 1],
    specular = [1, 1, 1, 1],
    roughness = 0.15,
    metallic = 0,
    f0 = 0.04
  );

  // Lighting parameters
  var L = [0.447735768366173, 0.497260947684137, 0.743144825477394];
  var Ambient = [0.1, 0.1, 0.1];
  var Diffuse = [0.8, 0.8, 0.8, 1];
  var Specular = [0.7, 0.7, 0.7, 1];
  var specularfactor = 3;

  // Material parameters
  var emissive = [0, 0, 0, 1];
  var ambient = [0, 0, 0, 1];
  var diffuse = [1, 0, 0, 1];
  var specular = [0.75, 0.75, 0.75, 1];
  var shininess = 0.5;

  var cameraPos = vec3.fromValues(0, 0, 2);
  var cameraLookAt = vec3.fromValues(0, 0, 0);
  var cameraUp = vec3.fromValues(1, 0, 0);
