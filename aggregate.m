function make_fractal_aggregate_metrics()
% Fractal aggregate (multi-field coupling)
% Roundness (R), Boxiness (B), 3D solidity (S3D), True sphericity (TS)

%% ----------------------- Generate parameters -----------------------
seed        = 1;
R0          = 0.5;
kStrength   = 0.3;      % Roughness strength
res         = 100;

% Hybrid noise
noiseScale  = 1.5;
H = struct();
H.s           = noiseScale;
H.sw          = 1.5*noiseScale;
H.eta         = 0.5;
H.octaves     = 5;
H.lacunarity  = 1.5;
H.gain        = 0.5;
H.alpha_fbm   = 0.8;
H.beta_ridged = 0.3;
H.gamma_chip  = 0.15;
H.chip_sharp  = 0.5;
H.enableLayer = true;
H.layerDir    = [1 1 1];
H.layerFreq   = 1.0;
H.layerAmp    = 0.15;
H.seed        = seed;


% Post-processing
enableTwist     = true;
enableDimControl= true;
targetDims      = [1 0.9 0.8];
fitMode         = 'pca';
fitType         = 'exact';
flattenBot      = true;
flatZ           = 0.0;
smoothIters     = 0;
albedoCol       = [0.55 0.55 0.55];

% Export STL
exportSTL   = true;
stlFilename = 'rock_metrics100.stl';   % Custom output filename

%% ----------------------- Generate intrinsic surface -----------------------
rng(seed);
[XS, YS, ZS] = sphere(res);
len = sqrt(XS.^2 + YS.^2 + ZS.^2) + eps;
ux = XS./len; uy = YS./len; uz = ZS./len;

N = hybrid_noise3_on_sphere(ux, uy, uz, H);
N = max(-1, min(1, N));
r = R0 * (1 + kStrength * N);

X = r.*ux; Y = r.*uy; Z = r.*uz;
[F_intr, V_intr] = surf2patch(X, Y, Z, N, 'triangles');

%% ----------------------- Post-processing -------------------
F = F_intr; V = V_intr;

if enableTwist
    twist.axis      = [1 0 1];
    twist.pivot     = [1 0 0];
    twist.rate_deg  = 45;
    V = twist_mesh(V, twist);
end
if smoothIters > 0
    [V(:,1), V(:,2), V(:,3)] = laplacian_smooth(V, F, smoothIters, 0.3);
end
if enableDimControl
    V = fit_mesh_to_box(V, targetDims, fitMode, fitType);
end
if flattenBot
    minZ = min(V(:,3));
    V(:,3) = V(:,3) - minZ + flatZ;
    z = V(:,3);
    bandH = max(0.015*range(z), 0.005);
    bandMask = (z >= flatZ) & (z < flatZ + bandH);
    t = (z(bandMask) - (flatZ+bandH)) / (flatZ - (flatZ+bandH));
    V(z<flatZ,3) = flatZ;
    V(bandMask,3) = V(bandMask,3).*(1-t) + flatZ.*t;
    [F, V] = cap_bottom_constrained(F, V, flatZ);  % rendering only
end

figure('Color','w'); ax=axes('Parent',gcf); hold(ax,'on');
p = patch('Faces',F,'Vertices',V,'FaceVertexCData',V(:,3), ...
          'FaceColor','interp','EdgeColor','none');
colormap(gray(256));
C = get(p,'FaceVertexCData'); if size(C,2)==1, C=repmat(C,1,3); end
C = normalize01(C);
C = 0.55*C + 0.45*repmat(albedoCol,size(C,1),1);
set(p,'FaceVertexCData',C,'FaceColor','interp');
axis equal off
camlight('headlight'); camlight('left'); lighting gouraud; material([0.3 0.6 0.1]);
view(45,25); zoom(1);
title(sprintf('Rock for Metrics | k=%.2f', kStrength));
exportgraphics(gcf,'rock_metrics100.png','Resolution',300,'BackgroundColor','none');

% ---------- Compute 4 metrics ----------
metrics = compute_shape_metrics(F, V);
fprintf('\n=== Shape Metrics (on processed mesh) ===\n');
fprintf('Roundness R          = %.6f   (1/r_in*(1/N*sum(r_i)))\n', metrics.Roundness);
fprintf('Boxiness B           = %.6f   (V_A / (l_a l_b l_c))\n', metrics.Boxiness);
fprintf('3D solidity S_3D     = %.6f   (V_A / V_conv)\n', metrics.S3D);
fprintf('True sphericity TS   = %.6f   (pi/S_A * ((6V_A)/pi)^(2/3))\n', metrics.TrueSphericity);

save('rock_metrics_results100.mat','metrics');

% ---------- Export STL (using post-processed F, V) ----------
if exportSTL
    write_stl_binary(stlFilename, F, V);
    fprintf('[STL] Exported binary STL: %s (%.0f triangles)\n', stlFilename, size(F,1));
end
end

%% ======================= Shape metrics computation =======================
function M = compute_shape_metrics(F, V)
% Volume and surface area
[V_A, C_vol] = volume_closed_mesh(F, V); %#ok<ASGLU>
S_A          = sum(tri_areas(V, F));

% Convex hull volume
[~, V_conv]  = convhulln(V, {'QJ'});  % QJ to avoid degeneracy warnings

% Principal axis lengths
[E, extents] = pca_obb(V);
l_a = extents(1); l_b = extents(2); l_c = extents(3);

% ===== Fitted inner sphere radius r_in =====
[c_fit, ~, ok_fit] = sphere_fit_ls(V);  % Global sphere fitting (all vertices)
if ~ok_fit
    c_fit = mean(V,1);                  
end
rad_all = sqrt(sum((V - c_fit).^2, 2));
r_in    = max(min(rad_all), eps);       

% ===== Corner radii r_i: dihedral angle criterion + one-ring local sphere fitting =====
r_list  = corner_radii_local_spheres_fast(F, V, 60);  % Default threshold 60°
r_list  = r_list(isfinite(r_list) & r_list > 0);

% Physical limit for corner sphere radius: should not exceed r_in
if ~isempty(r_list)
    r_list   = min(r_list, r_in);
    r_i_mean = mean(r_list);
else
    r_i_mean = 0;
end

% ===== Metrics =====
R   = r_i_mean / r_in;                         % Roundness
B   = V_A / (l_a*l_b*l_c);                     % Boxiness
S3D = V_A / max(V_conv, eps);                  % 3D solidity
TS  = (pi / max(S_A, eps)) * ((6*V_A)/pi)^(2/3); % True sphericity

M = struct('Roundness',R,'Boxiness',B,'S3D',S3D,'TrueSphericity',TS, ...
           'r_in',r_in,'r_i_mean',r_i_mean, ...
           'VA',V_A,'SA',S_A,'Vconv',V_conv, ...
           'la',l_a,'lb',l_b,'lc',l_c,'E',E,'c_fit',c_fit);
end

%% --------- Fast corner radii: dihedral angle threshold + one-ring fitting ----------
function r_list = corner_radii_local_spheres_fast(F, V, tauDeg)
% Fast corner detection + one-ring sphere fitting:
% - Corner: edge dihedral angle >= tauDeg
% - Each corner uses only the one-ring neighborhood for least-squares sphere fitting
if nargin < 3, tauDeg = 60; end

nV   = size(V,1);
adjV = vertex_adjacent_vertices(F, nV);

TR   = triangulation(F, V);
E    = edges(TR);
FE   = TR.edgeAttachments(E);

isCorner = false(nV,1);
for e = 1:size(E,1)
    faces = FE{e};
    if numel(faces) < 2
        theta = 180;  % Treat boundary as sharp
    else
        f1 = faces(1); f2 = faces(2);
        n1 = tri_normal(F(f1,:), V);
        n2 = tri_normal(F(f2,:), V);
        c  = max(-1, min(1, dot(n1,n2)));
        theta = acosd(c);
    end
    if theta >= tauDeg
        isCorner(E(e,1)) = true;
        isCorner(E(e,2)) = true;
    end
end

cornerIdx = find(isCorner);
r_list = nan(numel(cornerIdx),1);

for k = 1:numel(cornerIdx)
    v    = cornerIdx(k);
    ring = unique([v; adjV{v}(:)]);
    if numel(ring) < 4, continue; end
    P    = V(ring, :);
    [~, r, ok] = sphere_fit_ls(P);
    if ok && isfinite(r) && r > 0
        r_list(k) = r;
    end
end
end

%% --------- One-ring neighbors of each vertex ----------
function adj = vertex_adjacent_vertices(F, nV)
E = [F(:,[1 2]); F(:,[2 3]); F(:,[3 1])];
E = [E; E(:,[2 1])];
adj = cell(nV,1);
for i=1:size(E,1)
    adj{E(i,1)}(end+1) = E(i,2);
end
for i=1:nV, adj{i} = unique(adj{i}); end
end

%% --------- Algebraic least-squares sphere fitting (global/local) ----------
function [c, r, ok] = sphere_fit_ls(P)
% Least-squares sphere fit: x^2 + y^2 + z^2 + a x + b y + c z + d = 0
% Returns: sphere center c, radius r
X = P(:,1); Y = P(:,2); Z = P(:,3);
A = [X Y Z ones(size(X))];
b = -(X.^2 + Y.^2 + Z.^2);
lam = 1e-8;                             % Small regularization to avoid ill-conditioning
x = (A.'*A + lam*eye(4)) \ (A.'*b);
a = x(1); b2 = x(2); c2 = x(3); d = x(4);
c = -0.5*[a b2 c2];
r2 = sum(c.^2) - d;
ok = isfinite(r2) && r2 > 0;
if ok, r = sqrt(r2); else, r = NaN; end
end

%% --------- Triangle normal ----------
function n = tri_normal(f, V)
A = V(f(1),:); B = V(f(2),:); C = V(f(3),:);
n = cross(B-A, C-A); s = norm(n); 
if s>0, n = n/s; else, n=[0 0 1]; end
end

%% --------- Volume/centroid and surface area ----------
function [Vtot, C] = volume_closed_mesh(F, V)
% Tetrahedron summation w.r.t origin (assuming closed mesh)
V1 = V(F(:,1),:); V2 = V(F(:,2),:); V3 = V(F(:,3),:);
v6 = dot(V1, cross(V2, V3, 2), 2); % 6 times tetrahedron volume
V6 = sum(v6);
Vtot = abs(V6) / 6;

% Centroid from volume decomposition (signed), then volume-weighted
Cx = sum((V1(:,1) + V2(:,1) + V3(:,1)) .* v6) / (4*V6 + eps);
Cy = sum((V1(:,2) + V2(:,2) + V3(:,2)) .* v6) / (4*V6 + eps);
Cz = sum((V1(:,3) + V2(:,3) + V3(:,3)) .* v6) / (4*V6 + eps);
C  = [Cx Cy Cz];
end

function A = tri_areas(V,F)
u = V(F(:,2),:) - V(F(:,1),:);
v = V(F(:,3),:) - V(F(:,1),:);
A = 0.5*sqrt(sum(cross(u,v,2).^2,2));
A(A<=0) = eps;
end

%% --------- PCA ----------
function [E, extents] = pca_obb(V)
X = V - mean(V,1);
C = cov(X);
[E,D] = eig(C);
[~,idx] = sort(diag(D),'descend');
E = E(:,idx);             % Eigenvectors as principal axes (columns)
P = X * E;                % Projection onto principal axes
mins = min(P,[],1); maxs = max(P,[],1);
extents = sort(maxs - mins); % l_a <= l_b <= l_c
end

%% ======================= Noise, utilities, STL, etc. =======================
function N = hybrid_noise3_on_sphere(ux, uy, uz, P)
x0 = ux * P.sw; y0 = uy * P.sw; z0 = uz * P.sw;
w1 = fbm3_array(x0,         y0,          z0,          P.octaves, P.lacunarity, P.gain, P.seed+101);
w2 = fbm3_array(x0,         y0,          z0,          P.octaves, P.lacunarity, P.gain, P.seed+202);
w3 = fbm3_array(x0,         y0,          z0,          P.octaves, P.lacunarity, P.gain, P.seed+303);
x = ux * P.s + P.eta * w1;  y = uy * P.s + P.eta * w2;  z = uz * P.s + P.eta * w3;

N_fbm = fbm3_array(x, y, z, P.octaves, P.lacunarity, P.gain, P.seed+1);
N_rid = ridged_fbm3_array(x, y, z, P.octaves, P.lacunarity, P.gain, P.seed+2);

[F1, F2] = worley_F1F2_array(x, y, z, P.seed+3);
chips = (F2 - F1); chips = 1 - chips;
chips = max(0, chips).^P.chip_sharp;
chips = chips - mean(chips(:));

N = P.alpha_fbm*N_fbm + P.beta_ridged*N_rid + P.gamma_chip*chips;

if isfield(P,'enableLayer') && P.enableLayer
    L = P.layerAmp * sin(2*pi*P.layerFreq * (P.layerDir(1)*x + P.layerDir(2)*y + P.layerDir(3)*z));
    N = N .* (1 + L);
end
N = N / max(1e-8, max(abs(N(:))));
N = max(-1, min(1, N));
end

function N = ridged_fbm3_array(x,y,z,octaves,lacunarity,gain,seed)
perm = perlin_perm(seed);
N = zeros(size(x)); ampSum=0; freq=1.0; amp=1.0;
for o=1:octaves
    n = perlin3_array(x*freq, y*freq, z*freq, perm);
    n = 1 - abs(n);
    n = n.^2;
    N = N + amp*n;
    ampSum = ampSum + amp;
    freq = freq*lacunarity; amp = amp*gain;
end
N = N / max(ampSum, eps);
N = 2*N - 1;
end

function [F1, F2] = worley_F1F2_array(x, y, z, seed)
rng(seed);
X0=floor(x); Y0=floor(y); Z0=floor(z);
xf=x-X0; yf=y-Y0; zf=z-Z0;

F1 = inf(size(x)); F2 = inf(size(x));
for dz=-1:1
  for dy=-1:1
    for dx=-1:1
        Xi = X0 + dx; Yi = Y0 + dy; Zi = Z0 + dz;
        [jx,jy,jz] = jitter3(Xi, Yi, Zi, seed);
        px = dx + jx - xf;  py = dy + jy - yf;  pz = dz + jz - zf;
        d2 = px.*px + py.*py + pz.*pz;
        swap = d2 < F1;
        F2 = min(F2, max(F1, d2));
        F1(swap) = d2(swap);
    end
  end
end
F1 = sqrt(F1); F2 = sqrt(F2);
F1 = min(1, F1); F2 = min(1, F2);
end

function [jx,jy,jz] = jitter3(ix, iy, iz, seed)
ix = uint32(ix); iy = uint32(iy); iz = uint32(iz);
h = uint32(73856093).*ix + uint32(19349663).*iy + uint32(83492791).*iz + uint32(seed);
h = bitxor(h, bitshift(h,13));
h = h .* uint32(1274126177);
a = double(bitand(h, uint32(65535))) / 65536.0;
b = double(bitand(bitshift(h,-16), uint32(65535))) / 65536.0;
c = double(bitand(bitshift(h,-8),  uint32(65535))) / 65536.0;
jx=a; jy=b; jz=c;
end

function N = fbm3_array(x, y, z, octaves, lacunarity, gain, seed)
baseAmp=1.0; N=zeros(size(x)); ampSum=0; freq=1.0; amp=baseAmp;
perm = perlin_perm(seed);
for o=1:octaves
    N = N + amp * perlin3_array(x*freq, y*freq, z*freq, perm);
    ampSum = ampSum + amp; freq=freq*lacunarity; amp=amp*gain;
end
N = N / max(ampSum, eps);
N = max(-1, min(1, N));
end

function Pm = perlin_perm(seed)
rng(seed); p = randperm(256) - 1; Pm = int32([p p]);
end

function n = perlin3_array(x, y, z, perm)
n=zeros(size(x)); numelAll=numel(x);
for i=1:numelAll
    xi=x(i); yi=y(i); zi=z(i);
    X0=floor(xi); Y0=floor(yi); Z0=floor(zi);
    xf=xi-X0; yf=yi-Y0; zf=zi-Z0;
    X=mod(int32(X0),256); Y=mod(int32(Y0),256); Z=mod(int32(Z0),256);
    u=fade(xf); v=fade(yf); w=fade(zf);
    A=perm(X+1)+Y; AA=perm(mod(A,256)+1)+Z; AB=perm(mod(A+1,256)+1)+Z;
    B=perm(mod(X+1,256)+1)+Y; BA=perm(mod(B,256)+1)+Z; BB=perm(mod(B+1,256)+1)+Z;
    g000=grad(perm(mod(AA,256)+1), xf,  yf,  zf);
    g100=grad(perm(mod(BA,256)+1), xf-1,yf,  zf);
    g010=grad(perm(mod(AB,256)+1), xf,  yf-1,zf);
    g110=grad(perm(mod(BB,256)+1), xf-1,yf-1,zf);
    g001=grad(perm(mod(AA+1,256)+1), xf,  yf,  zf-1);
    g101=grad(perm(mod(BA+1,256)+1), xf-1,yf,  zf-1);
    g011=grad(perm(mod(AB+1,256)+1), xf,  yf-1,zf-1);
    g111=grad(perm(mod(BB+1,256)+1), xf-1,yf-1,zf-1);
    x1=lerp(g000,g100,u); x2=lerp(g010,g110,u); y1=lerp(x1,x2,v);
    x3=lerp(g001,g101,u); x4=lerp(g011,g111,u); y2=lerp(x3,x4,v);
    n(i)=lerp(y1,y2,w);
end
n = max(-1, min(1, n*0.95));
end

function t = fade(t), t=((6*t - 15).*t + 10).*t.^3; end
function a = lerp(a0,a1,t), a = a0 + t.*(a1 - a0); end
function g = grad(h,x,y,z)
h = bitand(h,15);
u = ifelse(h<8,x,y);
v = ifelse(h<4,y, ifelse(h==12 | h==14, x, z));
g = ((bitand(h,1)==0).*u + (bitand(h,1)==1).*(-u)) + ...
    ((bitand(h,2)==0).*v + (bitand(h,2)==2).*(-v));
end
function r = ifelse(cond,a,b), r=a; r(~cond)=b(~cond); end

function M = normalize01(M)
mn = min(M,[],1); mx = max(M,[],1); den=(mx-mn); den(den==0)=1;
M = (M - mn) ./ den;
end

function [x,y,z] = laplacian_smooth(V,F,iter,lambda)
n=size(V,1); A=sparse(n,n);
idx=[F(:,1) F(:,2); F(:,2) F(:,3); F(:,3) F(:,1)];
A = A + sparse(idx(:,1),idx(:,2),1,n,n);
A = A + sparse(idx(:,2),idx(:,1),1,n,n);
deg = sum(A,2); Dinv = spdiags(1./max(deg,1),0,n,n); W = Dinv*A;
P=V; for k=1:iter, P = P + lambda*(W*P - P); end
x=P(:,1); y=P(:,2); z=P(:,3);
end

function V2 = fit_mesh_to_box(V, targetDims, fitMode, fitType)
targetDims = targetDims(:)';  assert(numel(targetDims)==3);
switch lower(fitMode)
    case 'pca'
        mu = mean(V,1); X = V - mu; C = cov(X);
        [E,D] = eig(C); [~,idx] = sort(diag(D),'descend'); R = E(:,idx);
        P = X * R;
        ext = [max(P)-min(P)]; ext(ext==0)=1e-8;
        switch lower(fitType)
            case 'exact',   s = targetDims ./ ext;
            case 'uniform', s = min(targetDims ./ ext) * [1 1 1];
            otherwise, error('Unknown fitType');
        end
        V = (P * diag(s)) * R' + mu;
    case 'aabb'
        bb = bounding_box_axes(V, eye(3));
        ext = [bb.xmax-bb.xmin, bb.ymax-bb.ymin, bb.zmax-bb.zmin]; ext(ext==0)=1e-8;
        switch lower(fitType)
            case 'exact',   s = targetDims ./ ext;
            case 'uniform', s = min(targetDims ./ ext) * [1 1 1];
            otherwise, error('Unknown fitMode');
        end
        V = V .* s;
    otherwise, error('Unknown fitMode');
end
 V2 = V;
end

function bb = bounding_box_axes(V, R)
P = (V - mean(V,1)) * R;
bb.xmin=min(P(:,1)); bb.xmax=max(P(:,1));
bb.ymin=min(P(:,2)); bb.ymax=max(P(:,2));
bb.zmin=min(P(:,3)); bb.zmax=max(P(:,3));
end

function [Fout, Vout] = cap_bottom_constrained(F, V, flatZ)
tol = 1e-5;
TR  = triangulation(F, V);
Eb  = freeBoundary(TR);
onZ = @(id) abs(V(id,3) - flatZ) < tol;
mask = onZ(Eb(:,1)) & onZ(Eb(:,2));
E2   = Eb(mask,:);
if isempty(E2), Fout=F; Vout=V; return; end
loops = order_multiple_edge_loops(E2);
Fcap_all = [];
for li = 1:numel(loops)
    loop = loops{li};
    if numel(loop) < 3, continue; end
    P2 = V(loop,1:2);
    m  = numel(loop);
    C  = [(1:m)' [2:m 1]'];
    dt = delaunayTriangulation(P2, C);
    io = isInterior(dt);
    T  = dt.ConnectivityList(io,:);
    if isempty(T), continue; end
    Tg = loop(T);
    Fcap_all = [Fcap_all; Tg]; 
end
if isempty(Fcap_all), Fout=F; Vout=V; return; end
Fout = [F; Fcap_all];
Vout = V;
end

function loops = order_multiple_edge_loops(E)
G = containers.Map('KeyType','int32','ValueType','any');
for i=1:size(E,1)
    a=int32(E(i,1)); b=int32(E(i,2));
    if ~isKey(G,a), G(a)=int32([]); end
    if ~isKey(G,b), G(b)=int32([]); end
    G(a) = [G(a) b];
    G(b) = [G(b) a];
end
visited = containers.Map('KeyType','int32','ValueType','logical');
keysA   = cell2mat(G.keys);
loops   = {};
for s=1:numel(keysA)
    start = keysA(s);
    if isKey(visited,start) && visited(start), continue; end
    cur=int32(start); prev=int32(0); loop=double(cur);
    for it=1:500000
        visited(cur)=true; nbrs=G(cur);
        if isempty(nbrs), break; end
        if numel(nbrs)==1
            nxt=nbrs(1);
        else
            if prev==0, nxt=nbrs(1);
            else
                cand=nbrs(nbrs~=prev);
                if isempty(cand), break; end
                nxt=cand(1);
            end
        end
        if nxt==start, break; end
        loop(end+1)=double(nxt); prev=cur; cur=nxt;
        if numel(loop) > size(E,1)+5, break; end
    end
    loop = unique(loop,'stable');
    if numel(loop)>=3, loops{end+1}=loop; end
end
if isempty(loops), loops={}; end
end

function V2 = twist_mesh(V, T)
a = T.axis(:) / norm(T.axis);
p0 = T.pivot(:);
rate = deg2rad(T.rate_deg);
P = V' - p0;
s = a' * P;
theta = rate * s;
Ppar  = a * s;
Pperp = P - Ppar;
axP  = cross(a*ones(1,size(P,2)), Pperp, 1);
ct = cos(theta); st = sin(theta);
Pperp_rot = Pperp .* ct + axP .* st;
Prot = Ppar + Pperp_rot;
V2 = (Prot + p0)';
end

function write_stl_binary(filename, F, V)
    arguments
        filename (1,:) char
        F double
        V double
    end
    Vsingle = single(V);
    F = uint32(F);
    A = Vsingle(F(:,2),:) - Vsingle(F(:,1),:);
    B = Vsingle(F(:,3),:) - Vsingle(F(:,1),:);
    N = cross(A,B,2);
    L = sqrt(sum(N.^2,2)); L(L==0) = 1;
    N = N ./ L;

    fid = fopen(filename,'w','ieee-le');
    assert(fid>0, 'Cannot open STL file for writing: %s', filename);
    header = uint8(zeros(1,80));
    tag = sprintf('Metrics Rock | %s', datestr(now,'yyyy-mm-dd HH:MM:SS'));
    header(1:min(80,numel(tag))) = uint8(tag(1:min(80,numel(tag))));
    fwrite(fid, header, 'uint8');

    nF = uint32(size(F,1));
    fwrite(fid, nF, 'uint32');
    zeroAttr = uint16(0);
    for i = 1:double(nF)
        fwrite(fid, N(i,:), 'float32');
        fwrite(fid, Vsingle(F(i,1),:), 'float32');
        fwrite(fid, Vsingle(F(i,2),:), 'float32');
        fwrite(fid, Vsingle(F(i,3),:), 'float32');
        fwrite(fid, zeroAttr, 'uint16');
    end

    fclose(fid);
end
