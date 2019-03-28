V = zeros(100,100,100,'single');
V(50,50,50) = 1;

%% van Herk/Gil-Werman
doDilate = true; % false for erosion
Dirs = [1 0 0;
        0 1 0;
        0 0 1];
StrelSizes = [51,61,71];
BlockSize = [256,256,256]; % Generally a good block size
verbose = false;
Vo1 = mex_gorpho_flatLinearDilateErode3d_block(V,doDilate,StrelSizes,...
    Dirs,BlockSize,verbose);

%%
figure(1);
slice(Vo1,50,50,50);

%% Naive
Strel = strel('sphere',35);
d_Strel = cast(Strel.Neighborhood,class(V));
% Since the code performs general dilation / erosion we need to convert the
% strel. We turn 0 into -inf and 1 into 0.
d_Strel(d_Strel == 0) = -inf;
d_Strel(d_Strel == 1) = 0;
d_Strel = gpuArray(d_Strel);

doDilate = true; % false for erosion
BlockSize = [256,256,256]; % Generally a good block size
verbose = false;

Vo2 = mex_gorpho_genDilateErode3d_block(V,d_Strel,doDilate,BlockSize,...
    verbose);

%%
figure(2);
slice(Vo2,50,50,50);

%% Zonotope decomp
addpath ../zono_decomp/

A = StrelSphereDecomp(45);
Vo3 = DilateDecomp(V,A);

%%
figure(3);
slice(Vo3,50,50,50);