% MEX_GORPHO_FLATLINEARDILATEERODE3D_BLOCK
% Vo = mex_gorpho_flatLinearDilateErode3d_block(V,doDilate,StrelSizes,...
%     Dirs,BlockSize,verbose)
%
% INPUT
% V          - Input volume
% doDilate   - Dilate = true, Erosion = false
% StrelSizes - N vector with lengths for each line segment
% Dirs       - N x 3 matrix of step vectors for each line segment
% BlockSize  - Block size for block processing ([256,256,256] usually works
%              well)
% verbose    - Whether to print progess
%
% OUTPUT
% Vo - Output volume
%
% Patrick M. Jensen, 2019, Technical University of Denmark

% See source for mex file