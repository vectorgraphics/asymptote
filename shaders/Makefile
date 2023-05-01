shaders: material_vert material_frag

material_vert: material.vert
	glslangValidator -V material.vert -o material.vert.spv -S vert

material_frag: material.frag
	glslangValidator -V material.frag -o material.frag.spv -S frag