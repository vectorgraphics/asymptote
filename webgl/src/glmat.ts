// This module re-exports gl-matrix functions. This is useful for
// animations or any additional code.

// In javascript, gl-matrix functions is accessed by glmat variable.

import * as glmat from "gl-matrix";

globalThis.glmat=glmat;
