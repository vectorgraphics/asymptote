shaders: material color point triangle transparent blend partials

material:
	python3 genShaders.py material MATERIAL NORMAL HAVE_INTERLOCK HAVE_SSBO GPUINDEXING

	glslangValidator -V material.vert.glsl -o material.vert.spv -S vert
	glslangValidator -V material.frag.glsl -o material.frag.spv -S frag

color:
	python3 genShaders.py color MATERIAL COLOR NORMAL HAVE_INTERLOCK HAVE_SSBO GPUINDEXING

	glslangValidator -V color.vert.glsl -o color.vert.spv -S vert
	glslangValidator -V color.frag.glsl -o color.frag.spv -S frag

triangle:
	python3 genShaders.py triangle MATERIAL COLOR NORMAL GENERAL HAVE_INTERLOCK HAVE_SSBO GPUINDEXING

	glslangValidator -V triangle.vert.glsl -o triangle.vert.spv -S vert
	glslangValidator -V triangle.frag.glsl -o triangle.frag.spv -S frag

point:
	python3 genShaders.py point MATERIAL NOLIGHTS WIDTH HAVE_INTERLOCK HAVE_SSBO GPUINDEXING

	glslangValidator -V point.vert.glsl -o point.vert.spv -S vert
	glslangValidator -V point.frag.glsl -o point.frag.spv -S frag

transparent:
	python3 genShaders.py transparent MATERIAL COLOR NORMAL TRANSPARENT HAVE_SSBO GPUINDEXING HAVE_INTERLOCK

	glslangValidator -V transparent.vert.glsl -o transparent.vert.spv -S vert
	glslangValidator -V transparent.frag.glsl -o transparent.frag.spv -S frag

blend:

	glslangValidator -V blend.vert.glsl -o blend.vert.spv -S vert
	glslangValidator -V blend.frag.glsl -o blend.frag.spv -S frag

partials:

	glslangValidator -V count.frag.glsl -o count.frag.spv -S frag
	glslangValidator -V sum1.comp.glsl -o sum1.comp.spv -S comp
	glslangValidator -V sum2.comp.glsl -o sum2.comp.spv -S comp
	glslangValidator -V sum3.comp.glsl -o sum3.comp.spv -S comp
