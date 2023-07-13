shaders: material color point triangle transparent blend partials

material:
	python3 genShaders.py base materialOpaque MATERIAL NORMAL OPAQUE
	python3 genShaders.py base materialCount MATERIAL NORMAL
	python3 genShaders.py base materialIndexing MATERIAL NORMAL GPUINDEXING
	python3 genShaders.py base materialIndexingSSBO MATERIAL NORMAL GPUINDEXING HAVE_SSBO

color:
	python3 genShaders.py base colorOpaque MATERIAL COLOR NORMAL OPAQUE
	python3 genShaders.py base colorCount MATERIAL COLOR NORMAL
	python3 genShaders.py base colorIndexing MATERIAL COLOR NORMAL GPUINDEXING
	python3 genShaders.py base colorIndexingSSBO MATERIAL NORMAL GPUINDEXING HAVE_SSBO

triangle:
	python3 genShaders.py base triangleOpaque MATERIAL COLOR NORMAL GENERAL OPAQUE
	python3 genShaders.py base triangleCount MATERIAL COLOR NORMAL GENERAL
	python3 genShaders.py base triangleIndexing MATERIAL COLOR NORMAL GENERAL GPUINDEXING
	python3 genShaders.py base triangleIndexingSSBO MATERIAL COLOR NORMAL GENERAL GPUINDEXING HAVE_SSBO

point:
	python3 genShaders.py base pointOpaque MATERIAL NOLIGHTS WIDTH OPAQUE
	python3 genShaders.py base pointCount MATERIAL NOLIGHTS WIDTH
	python3 genShaders.py base pointIndexing MATERIAL NOLIGHTS WIDTH GPUINDEXING
	python3 genShaders.py base pointIndexingSSBO MATERIAL NOLIGHTS WIDTH GPUINDEXING HAVE_SSBO

transparent:
	python3 genShaders.py base transparentCount MATERIAL COLOR NORMAL GENERAL
	python3 genShaders.py base transparentIndexing MATERIAL COLOR NORMAL GENERAL TRANSPARENT GPUINDEXING
	python3 genShaders.py base transparentIndexingSSBO MATERIAL COLOR NORMAL GENERAL TRANSPARENT GPUINDEXING HAVE_SSBO

blend:
	python3 genShaders.py blend blendIndexing GPUINDEXING
	python3 genShaders.py blend blendIndexingSSBO GPUINDEXING HAVE_SSBO

partials:

	glslangValidator -V count.frag.glsl -o count.frag.spv -S frag
	glslangValidator -V sum1.comp.glsl -o sum1.comp.spv -S comp
	glslangValidator -V sum2.comp.glsl -o sum2.comp.spv -S comp
	glslangValidator -V sum3.comp.glsl -o sum3.comp.spv -S comp
