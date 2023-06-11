shaders: material_vert material_frag point_vert point_frag compute_comp

material_vert: material.vert.glsl
	glslangValidator -V material.vert.glsl -o material.vert.spv -S vert

material_frag: material.frag.glsl
	glslangValidator -V material.frag.glsl -o material.frag.spv -S frag

point_vert: point.vert.glsl
	glslangValidator -V point.vert.glsl -o point.vert.spv -S vert

point_frag: point.frag.glsl
	glslangValidator -V point.frag.glsl -o point.frag.spv -S frag

compute_comp: compute.comp
	glslangValidator -V compute.comp -o compute.comp.spv -S comp