function Vo = ErodeDecomp(V,A,BlockSize)
% ERODEDECOMP Erotion with zonohedral approximation
% Vo = ErodeDecomp(V,A,BlockSize)
%
% INPUT
% V         - Volume to erode
% A         - Coefficients for zonohedral approximation
% BlockSize - (Optional, default [256,256,256]) Block size for block
%             processing
% verbose   - (Optional, default: false) Print progress
%
% OUTPUT
% Vo - Eroded volume
%
% See also: STRELSPHEREDECOMP
%
% Patrick M. Jensen, 2019, Technical University of Denmark

if nargin < 3, BlockSize = [256,256,256]; end
if nargin < 4, verbose = false; end
Dirs = [1 0 0;
        0 -1 0;
        0 0 1;
        1 1 0;
        -1 1 0;
        -1 0 -1;
        1 0 -1;
        0 1 1;
        0 -1 1;
        -1 -1 -1;
        1 1 -1;
        1 -1 1;
        -1 1 1];

StrelSizs = [ones(1,3)*A(1), ones(1,6)*A(2), ones(1,4)*A(3)];
   
Vo = mex_gorpho_flatLinearDilateErode3d_block(V,false,StrelSizs,Dirs,...
    BlockSize, verbose);