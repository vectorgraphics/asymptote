import graph3;
import solids;

if(settings.outformat == "")
  settings.outformat="html";

size(20cm);

currentprojection=  perspective(
    camera=(-10.32483,10.32483,10.47533),
    up=(0.009815727,-0.009815727,0.01963145),
    target=(-0.32483398445800127,0.32483398445800127,0.47533398445800046),
    zoom=0.48101709809096993,
    angle=33.996826322482754,
    autoadjust=true);

currentlight = light(
    background=white,
    diffuse=new pen[] {gray(0.8), gray(0.8), gray(0.7)},
    specular=new pen[] {gray(0.1), gray(0.1), gray(0.1)},
    position=new triple[] {currentprojection.camera, (-5,-5,5), (0,0,10)}
);

// ===============================
// Parameters
// ===============================
real L1_Len = 2;
real L2_Len = 2;
real L3_Len = 2;

bool staticImage = (settings.outformat != "html");

// =========================================================
// JAVASCRIPT BRIDGE (Modified for Individual Sliders)
// =========================================================
javascript("
// ===============================
// 1. Storage for joint angles
// ===============================
window.jointStates = { q1: 0, q2: 0, q3: 0 };

// ===============================
// 2. Matrix Math Helpers
// ===============================
function multiply(A, B) {
    let C = new Array(16).fill(0);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) { sum += A[row + k*4] * B[k + col*4]; }
            C[row + col*4] = sum;
        }
    }
    return C;
}

function apply(M, p) {
    return [
        M[0]*p[0] + M[4]*p[1] + M[8]*p[2]  + M[12],
        M[1]*p[0] + M[5]*p[1] + M[9]*p[2]  + M[13],
        M[2]*p[0] + M[6]*p[1] + M[10]*p[2] + M[14]
    ];
}

function getRotationZ(deg) {
    let rad = deg * Math.PI / 180;
    let c = Math.cos(rad), s = Math.sin(rad);
    return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

// ===============================
// 3. Transform Functions
// ===============================
window.J1 = function(p, t) {
    if(!window.L0_OFFSET) return p;
    return apply(multiply(window.L0_OFFSET, getRotationZ(window.jointStates.q1)), p);
};

window.J2 = function(p, t) {
    if(!window.L1_OFFSET) return p;
    return apply(multiply(window.L1_OFFSET, getRotationZ(window.jointStates.q2)), p);
};

window.J3 = function(p, t) {
    if(!window.L2_OFFSET) return p;
    return apply(multiply(window.L2_OFFSET, getRotationZ(window.jointStates.q3)), p);
};

// ===============================
// 4. UI + Responsive Styling
// ===============================
window.addEventListener('load', function(){

    // Hide default controls
    let hide = document.createElement('style');
    hide.textContent = '.asy-player-controls, #asy-slider { display: none !important; }';
    document.head.appendChild(hide);

    // ----------------------
    // Responsive Slider CSS
    // ----------------------
    let style = document.createElement('style');
    style.textContent = `
    #control-panel {
        width: min(300px, 80vw);
    }

    .control-slider {
        width: 100%;
        height: 4vh;
    }

    .control-slider::-webkit-slider-runnable-track {
        height: 1.2vh;
        background: #ddd;
        border-radius: 0.6vh;
    }

    .control-slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 4vh;
        height: 4vh;
        margin-top: -1.4vh;
        border-radius: 50%;
        background: #4CAF50;
    }

    .control-slider::-moz-range-track { height: 1.2vh; background: #ddd; }
    .control-slider::-moz-range-thumb {
        width: 4vh;
        height: 4vh;
        border-radius: 50%;
        background: #4CAF50;
    }
    `;
    document.head.appendChild(style);

    // ----------------------
    // Control Panel
    // ----------------------
    let panel = document.createElement('div');
    panel.id = 'control-panel';
    panel.style.cssText = `
        position:fixed;
        top:2vh;
        left:2vw;
        z-index:9999;
        background:rgba(255,255,255,0.9);
        padding:2vh;
        border-radius:10px;
        font-family:sans-serif;
        box-shadow:3px 3px 10px rgba(0,0,0,0.2);
    `;

    panel.innerHTML = '<b style=\"display:block;margin-bottom:10px;border-bottom:1px solid #ccc\">Joint Controls</b>';

    ['q1','q2','q3'].forEach(joint => {
        let div = document.createElement('div');
        div.style.marginBottom = '10px';

        div.innerHTML =
            `<label style=\"width:30px;display:inline-block\">${joint.toUpperCase()}:</label>` +
            `<input type=\"range\" class=\"control-slider\" min=\"-180\" max=\"180\" value=\"0\" id=\"slider-${joint}\">` +
            `<span id=\"val-${joint}\" style=\"margin-left:8px\">0°</span>`;

        let slider = div.querySelector('input');
        slider.oninput = function() {
            window.jointStates[joint] = parseFloat(this.value);
            document.getElementById('val-'+joint).innerText = this.value + '°';
            if (window.updateScene) requestAnimationFrame(window.updateScene);
        };

        panel.appendChild(div);
    });

    document.body.appendChild(panel);

    // ----------------------
    // Title (Top Center)
    // ----------------------
    let title = document.createElement('div');
    title.innerHTML = '<b>RRR Three-Joint Robot Arm</b>';
    title.style.cssText = `
        position:fixed;
        top:10px;
        left:50%;
        transform:translateX(-50%);
        font-size:20px;
        font-family:Arial;
        color:black;
        z-index:10000;
        pointer-events:none;
    `;
    document.body.appendChild(title);

    // ----------------------
    // Footer (Bottom Right)
    // ----------------------
    let footer = document.createElement('div');
    footer.innerHTML = 'Powered by <tt>Asymptote</tt> & <tt>WebGL</tt>';
    footer.style.cssText = `
        position:fixed;
        bottom:10px;
        right:15px;
        font-size:12px;
        font-family:Arial;
        color:darkblue;
        z-index:10000;
        pointer-events:none;
    `;
    document.body.appendChild(footer);

});
");

// ===============================
// ASYMPTOTE GEOMETRY & ASSEMBLY
// (UNCHANGED PER YOUR REQUEST)
// ===============================
void joint_r1_f(transform3 T) {
    real R = 0.35, H = 0.5;
    pen surfacepen = lightgray;
    pen edgepen = black + linewidth(0.5mm);
    triple axis = unit(T*Z - T*O);
    revolution r = cylinder(T*(-0.5*H*Z), R, H, T*Z - T*O);
    draw(surface(r), surfacepen);
    draw(surface(circle(T*(0,0, H/2), R, axis)), surfacepen);
    draw(circle(T*(0,0, H/2), R, axis), edgepen);
    draw(surface(circle(T*(0,0,-H/2), R, axis)), surfacepen);
    draw(circle(T*(0,0,-H/2), R, axis), edgepen);
}

transform3 joint_r1_m(transform3 T, real angle) {
    transform3 R = rotate(angle, Z);
    transform3 Tm = T * R;
    pen movepen = lightblue;
    surface box = shift(-0.2*(X+Y) - 0.3*Z) * scale(0.4, 0.4, 0.6) * unitcube;
    draw(Tm*box, movepen);
    dot(Tm*(0.2,0,0.3), red);
    dot(Tm*(0.2,0,-0.3), red);
    return Tm;
}

void joint_r2_f(transform3 T) {
    real R = 0.35, H = 0.5;
    pen surfacepen = gray(0.85);
    pen edgepen = black + linewidth(0.5mm);
    triple axis = unit(T*Z - T*O);
    revolution r = cylinder(T*(-0.5*H*Z), R, H, axis);
    draw(surface(r), surfacepen);
    draw(surface(circle(T*(0,0, H/2), R, axis)), surfacepen);
    draw(circle(T*(0,0, H/2), R, axis), edgepen);
    draw(surface(circle(T*(0,0,-H/2), R, axis)), surfacepen);
    draw(circle(T*(0,0,-H/2), R, axis), edgepen);
}

transform3 joint_r2_m(transform3 T, real angle) {
    pen movepen = lightgreen;
    pen edgepen = black + linewidth(0.5mm);
    transform3 R = rotate(angle, Z);
    transform3 Tm = T * R;
    surface box = shift(-0.2*(X+Y) - 0.3*Z) * scale(0.4, 0.4, 0.6) * unitcube;
    draw(Tm*box, movepen);
    path3 p1 = (0,0,0.3)--(0,0,0.35)--(0.4,0,0.35)--(0.4,0,-0.35)--(0,0,-0.35)--(0,0,-0.3);
    draw(Tm*p1, edgepen);
    dot(Tm*(0.2,0,0.3), red);
    dot(Tm*(0.2,0,-0.3), red);
    return Tm * rotate(90, Y);
}

void draw_gripper_simple(transform3 T) {
    pen p = black + linewidth(1.2);
    real w = 0.1, h = 0.2;
    draw(T*(-w, 0, 0) -- T*(-w, 0, h), p);
    draw(T*( w, 0, 0) -- T*( w, 0, h), p);
    draw(T*(-w, 0, 0) -- T*( w, 0, 0), p);
}

void attach_coordinates(transform3 T, real length=2, string suffix="") {
    draw(T*scale3(length)*(O--X), red, Arrow3(size=10));
    draw(T*scale3(length)*(O--Y), green, Arrow3(size=10));
    draw(T*scale3(length)*(O--Z), blue, Arrow3(size=10));
    string sub = (suffix == "") ? "" : "_{" + suffix + "}";
    label("$x" + sub + "$", T*scale3(length)*(1.2*X), T*X, red);
    label("$y" + sub + "$", T*scale3(length)*(1.2*Y), T*Y, green);
    label("$z" + sub + "$", T*scale3(length)*(1.2*Z), T*Z, blue);
}

void exportToJS(string name, transform3 T) {
    string jsCommand = "window." + name + " = [" +
        string(T[0][0]) + "," + string(T[1][0]) + "," + string(T[2][0]) + "," + string(T[3][0]) + "," +
        string(T[0][1]) + "," + string(T[1][1]) + "," + string(T[2][1]) + "," + string(T[3][1]) + "," +
        string(T[0][2]) + "," + string(T[1][2]) + "," + string(T[2][2]) + "," + string(T[3][2]) + "," +
        string(T[0][3]) + "," + string(T[1][3]) + "," + string(T[2][3]) + "," + string(T[3][3]) + "];";
    javascript(jsCommand);
}

// ===============================
// World Frame & Floor
// ===============================
draw(O--X, red, Arrow3);
draw(O--Y, green, Arrow3);
draw(O--Z, blue, Arrow3);
label("$x$", 1.1*X, N, red);
label("$y$", 1.1*Y, N, green);
label("$z$", 1.0*Z, E, blue);

pen bg=gray(0.9); real r = 2.5;
draw(surface((r,r,0)--(-r,r,0)--(-r,-r,0)--(r,-r,0)--cycle),bg,bg,light=nolight);

transform3 L0 = shift(0, 0, 2);
draw(O--L0*O, linewidth(2));
joint_r1_f(L0);
attach_coordinates(L0*shift(0,0,2), 1.5, "1");

// ===============================
// Shared arm-drawing code
// Lbase is the input transform; returns output transform for chaining.
// joint1 returns L1_c, joint2 returns L2_b, joint3 returns nothing.
// ===============================
transform3 drawJoint1(transform3 Lbase) {
    transform3 L1_a = joint_r1_m(Lbase, 0);
    attach_coordinates(L1_a*shift(0,0,2)*rotate(90,X), 1.3, "2");
    transform3 L1_b = L1_a * shift(0,0,L1_Len) * rotate(180, Z);
    draw(L1_a*O--L1_b*O, linewidth(2));
    transform3 L1_c = L1_b * rotate(-90, X);
    joint_r2_f(L1_c);
    return L1_c;
}

transform3 drawJoint2(transform3 Lbase) {
    transform3 L2_a = joint_r2_m(Lbase, 0);
    attach_coordinates(L2_a*rotate(-90,Z), 1, "3");
    transform3 L2_b = L2_a*shift(0,0,L2_Len);
    draw(L2_a*(0,0,0.4)--L2_b*O, linewidth(2));
    joint_r1_f(L2_b);
    return L2_b;
}

void drawJoint3(transform3 Lbase) {
    transform3 L3_a = joint_r1_m(Lbase, 0);
    transform3 L3_b = L3_a*shift(0,0,L3_Len);
    draw(L3_a*O--L3_b*O, linewidth(2));
    draw_gripper_simple(L3_b);
    attach_coordinates(L3_b*rotate(-90,Z), 0.5, "4");
}

// ===============================
// HTML mode: nested beginTransform blocks with JS animation
// Geometry is at identity; JS applies the transforms.
// ===============================
if(!staticImage) {
beginTransform("function(x,t){ return J1(x,t); }", 1);
    transform3 L1 = drawJoint1(identity(4));
    beginTransform("function(x,t){ return J2(x,t); }", 1);
        transform3 L2 = drawJoint2(identity(4));
        beginTransform("function(x,t){ return J3(x,t); }", 1);
            drawJoint3(identity(4));
        endTransform();
    endTransform();
endTransform();
exportToJS("L0_OFFSET", L0);
exportToJS("L1_OFFSET", L1);
exportToJS("L2_OFFSET", L2);
} // end !staticImage

// ===============================
// PDF mode: static geometry with pre-computed transforms
// ===============================
if(staticImage) {
    transform3 L1 = drawJoint1(L0);
    transform3 L2 = drawJoint2(L1);
    drawJoint3(L2);
}
