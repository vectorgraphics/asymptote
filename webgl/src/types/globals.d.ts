declare global {
    interface Document {
        asy: any;
    }

    interface Window {
        webGLStart: () => void,
        light: Function,
        material: Function,
        patch: Function,
        curve: Function,
        pixel: Function,
        triangles: Function,
        sphere: Function,
        disk: Function,
        cylinder: Function,
        tube: Function,
        Positions: any[],
        Normals: any[],
        Colors: any[],
        Indices: any[]
    }
}

export {};
