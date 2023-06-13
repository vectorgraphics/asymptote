shaders: material color point triangle

material:
	python3 genShaders.py material MATERIAL NORMAL

	glslangValidator -V material.vert.glsl -o material.vert.spv -S vert
	glslangValidator -V material.frag.glsl -o material.frag.spv -S frag

color:
	python3 genShaders.py color MATERIAL COLOR NORMAL

	glslangValidator -V color.vert.glsl -o color.vert.spv -S vert
	glslangValidator -V color.frag.glsl -o color.frag.spv -S frag

triangle:
	python3 genShaders.py triangle MATERIAL COLOR NORMAL GENERAL

	glslangValidator -V triangle.vert.glsl -o triangle.vert.spv -S vert
	glslangValidator -V triangle.frag.glsl -o triangle.frag.spv -S frag

point:
	python3 genShaders.py point MATERIAL NOLIGHTS WIDTH

	glslangValidator -V point.vert.glsl -o point.vert.spv -S vert
	glslangValidator -V point.frag.glsl -o point.frag.spv -S frag
