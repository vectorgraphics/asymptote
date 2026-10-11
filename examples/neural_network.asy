settings.outformat = "pdf";
settings.prc = false;
settings.render = 0;
unitsize(0.5cm);

real node_radius = 1;
int node_ix = 0;

struct Node {
    pair center;
    string label;
    int state;
    void operator init(pair center, int state){
        this.state = state;
        this.center = center;
        this.label = format("$x_{%d}$", node_ix);
        node_ix += 1;
    }

    void draw(){
        draw(circle(this.center, node_radius));
        if (state == 0){
            fill(circle(this.center, node_radius), white);
        } else {
            fill(circle(this.center, node_radius), red);
        }
        label(this.label, this.center);
    }
}

// ============= FEED FORWARD =============

// Coordinates for each node
int n_layers = 3;
int n_neurons = 5;
real height = 2.8;
real spacing = 2.8;
real[][] X = new real[n_layers][n_neurons];
Node[][] nodes = new Node[n_layers][n_neurons];

for (int i = 0; i < n_neurons; ++i){
    //Populating first coordinates
    X[0][i] = i*spacing;
    nodes[0][i] = Node((X[0][i], 0), 0);
}
for (int l = 1; l < n_layers; ++l){
    for (int i = 0; i < n_neurons - l; ++i){
        X[l][i] = (X[l-1][i] + X[l-1][i+1])/2;
        nodes[l][i] = Node((X[l][i], l*height), 0);
        for (int j = 0; j < n_neurons - l + 1; ++j){
            draw((X[l][i], l*height) -- (X[l-1][j], (l-1)*height));
            draw((X[l][i], l*height) -- (X[l-1][j], (l-1)*height));
        }
    }
}

// Separate loop to ensure drawing nodes on top of edges
for (int l = 0; l < n_layers; ++l){
    for (int i = 0; i < n_neurons - l; ++i){
        nodes[l][i].draw();
    }
}

// ============= MATRIX =============

void drawBrace(pair start, pair end) {
    real mid = (start.y + end.y) / 2;
    real len = end.y - start.y;
    real tip = 0.2; // width of the brace tip
    path p = (start.x, start.y) -- (start.x - tip, start.y) -- (start.x - tip, end.y) -- (start.x, end.y);
    draw(p);
}
real bracket_y = 0;
real bracket_spacing = 0.1;
real bracket_x = n_neurons*spacing;
for (int l = 0; l < n_layers; ++l){
    drawBrace((bracket_x,bracket_y + bracket_spacing - 0.2), (bracket_x,bracket_y + n_neurons - l - 0.5 - bracket_spacing));
    bracket_y = bracket_y + n_neurons - l;
}


real x_offset = n_neurons*spacing + 1;
real m_spacing = 2;
int T = 8;
for (int i = 0; i < node_ix; ++i){
    for (int t = 0; t < T; ++t){
        label("$x_{" + string(i) + "," + string(t) + "}$", (t*m_spacing + x_offset, i));
    }
}

draw((x_offset, -0.6) -- (x_offset + (T-1)*m_spacing, -0.6), arrow=Arrow);
label("Time", (0.5*(2*x_offset + T*m_spacing), -1.1));
