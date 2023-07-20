shaders:

material:
	python3 genShaders.py base materialOpaque MATERIAL NORMAL OPAQUE
	python3 genShaders.py base materialCount MATERIAL NORMAL
	python3 genShaders.py base materialCountCompress MATERIAL NORMAL GPUCOMPRESS
	python3 genShaders.py base materialDefault MATERIAL NORMAL HAVE_SSBO
	python3 genShaders.py base materialIndexing MATERIAL NORMAL GPUINDEXING
	python3 genShaders.py base materialIndexingSSBO MATERIAL NORMAL GPUINDEXING HAVE_SSBO
	python3 genShaders.py base materialIndexingSSBOInterlock MATERIAL NORMAL GPUINDEXING HAVE_SSBO HAVE_INTERLOCK
	python3 genShaders.py base materialIndexingSSBOInterlockCompress MATERIAL NORMAL GPUINDEXING HAVE_SSBO HAVE_INTERLOCK GPUCOMPRESS

color:
	python3 genShaders.py base colorOpaque MATERIAL COLOR NORMAL OPAQUE
	python3 genShaders.py base colorCount MATERIAL COLOR NORMAL
	python3 genShaders.py base colorCountCompress MATERIAL COLOR NORMAL GPUCOMPRESS
	python3 genShaders.py base colorDefault MATERIAL COLOR NORMAL HAVE_SSBO
	python3 genShaders.py base colorIndexing MATERIAL COLOR NORMAL GPUINDEXING
	python3 genShaders.py base colorIndexingSSBO MATERIAL NORMAL GPUINDEXING HAVE_SSBO
	python3 genShaders.py base colorIndexingSSBOInterlock MATERIAL NORMAL GPUINDEXING HAVE_SSBO HAVE_INTERLOCK
	python3 genShaders.py base colorIndexingSSBOInterlockCompress MATERIAL NORMAL GPUINDEXING HAVE_SSBO HAVE_INTERLOCK GPUCOMPRESS

triangle:
	python3 genShaders.py base triangleOpaque MATERIAL COLOR NORMAL GENERAL OPAQUE
	python3 genShaders.py base triangleCount MATERIAL COLOR NORMAL GENERAL
	python3 genShaders.py base triangleCountCompress MATERIAL COLOR NORMAL GENERAL GPUCOMPRESS
	python3 genShaders.py base triangleDefault MATERIAL COLOR NORMAL GENERAL HAVE_SSBO
	python3 genShaders.py base triangleIndexing MATERIAL COLOR NORMAL GENERAL GPUINDEXING
	python3 genShaders.py base triangleIndexingSSBO MATERIAL COLOR NORMAL GENERAL GPUINDEXING HAVE_SSBO
	python3 genShaders.py base triangleIndexingSSBOInterlock MATERIAL COLOR NORMAL GENERAL GPUINDEXING HAVE_SSBO HAVE_INTERLOCK
	python3 genShaders.py base triangleIndexingSSBOInterlockCompress MATERIAL COLOR NORMAL GENERAL GPUINDEXING HAVE_SSBO HAVE_INTERLOCK GPUCOMPRESS

point:
	python3 genShaders.py base pointOpaque MATERIAL NOLIGHTS WIDTH OPAQUE
	python3 genShaders.py base pointCount MATERIAL NOLIGHTS WIDTH
	python3 genShaders.py base pointCountCompress MATERIAL NOLIGHTS WIDTH GPUCOMPRESS
	python3 genShaders.py base pointDefault MATERIAL NOLIGHTS WIDTH HAVE_SSBO
	python3 genShaders.py base pointIndexing MATERIAL NOLIGHTS WIDTH GPUINDEXING
	python3 genShaders.py base pointIndexingSSBO MATERIAL NOLIGHTS WIDTH GPUINDEXING HAVE_SSBO
	python3 genShaders.py base pointIndexingSSBOInterlock MATERIAL NOLIGHTS WIDTH GPUINDEXING HAVE_SSBO HAVE_INTERLOCK
	python3 genShaders.py base pointIndexingSSBOInterlockCompress MATERIAL NOLIGHTS WIDTH GPUINDEXING HAVE_SSBO HAVE_INTERLOCK GPUCOMPRESS

transparent:
	python3 genShaders.py base transparentCount MATERIAL COLOR NORMAL TRANSPARENT GENERAL
	python3 genShaders.py base transparentCountCompress MATERIAL COLOR NORMAL TRANSPARENT GENERAL GPUCOMPRESS
	python3 genShaders.py base transparentDefault MATERIAL COLOR NORMAL GENERAL TRANSPARENT HAVE_SSBO
	python3 genShaders.py base transparentIndexing MATERIAL COLOR NORMAL GENERAL TRANSPARENT GPUINDEXING
	python3 genShaders.py base transparentIndexingSSBO MATERIAL COLOR NORMAL GENERAL TRANSPARENT GPUINDEXING HAVE_SSBO
	python3 genShaders.py base transparentIndexingSSBOInterlock MATERIAL COLOR NORMAL GENERAL TRANSPARENT GPUINDEXING HAVE_SSBO HAVE_INTERLOCK
	python3 genShaders.py base transparentIndexingSSBOInterlockCompress MATERIAL COLOR NORMAL GENERAL TRANSPARENT GPUINDEXING HAVE_SSBO HAVE_INTERLOCK GPUCOMPRESS

blend:
	python3 genShaders.py blend blendDefault HAVE_SSBO
	python3 genShaders.py blend blendIndexing GPUINDEXING
	python3 genShaders.py blend blendIndexingSSBO GPUINDEXING HAVE_SSBO
	python3 genShaders.py blend blendIndexingSSBOInterlock GPUINDEXING HAVE_INTERLOCK
	python3 genShaders.py blend blendIndexingSSBOInterlockCompress GPUINDEXING HAVE_INTERLOCK GPUCOMPRESS

compress:
	glslangValidator -V compress.frag.glsl -o compress.frag.spv -S frag
	glslangValidator -V compress.vert.glsl -o compress.vert.spv -S vert

partials:

	glslangValidator -V count.frag.glsl -o count.frag.spv -S frag
	glslangValidator -V countCompress.frag.glsl -o countCompress.frag.spv -S frag
	glslangValidator -V sum1.comp.glsl -o sum1.comp.spv -S comp
	glslangValidator -V sum2.comp.glsl -o sum2.comp.spv -S comp
	glslangValidator -V sum3.comp.glsl -o sum3.comp.spv -S comp
