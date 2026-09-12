// This module re-exports gl-matrix functions. This is useful for
// animations or any additional code.

// In javascript, gl-matrix functions is accessed by glmat variable.

import * as mat3 from "gl-matrix/mat3";
import * as mat4 from "gl-matrix/mat4";
import * as vec3 from "gl-matrix/vec3";
import * as vec4 from "gl-matrix/vec4";

globalThis.glmat = { mat3, mat4, vec3, vec4 };

