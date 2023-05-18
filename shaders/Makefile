shaders: material_vert material_frag compute_comp

material_vert: material.vert
	glslangValidator -V material.vert -o material.vert.spv -S vert

material_frag: material.frag
	glslangValidator -V material.frag -o material.frag.spv -S frag

compute_comp: compute.comp
	glslangValidator -V compute.comp -o compute.comp.spv -S comp