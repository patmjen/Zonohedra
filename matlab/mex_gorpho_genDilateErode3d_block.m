% MEX_GORPHO_GENDILATEERODE3D_BLOCK
% Vo = mex_gorpho_genDilateErode3d_block(V,d_Strel,doDilate,BlockSize,...
%     verbose)
%
% NOTE: This function does *general* grayscale morhology. Thus, a regular
% binary structuring element should be converted as follow:
%  1. set all zeros to -inf
%  2. set all ones to 0
%
% INPUT
% V          - Input volume (must be single or double)
% d_Strel    - gpuArray with structuring element
% doDilate   - Dilate = true, Erosion = false
% BlockSize  - Block size for block processing ([256,256,256] usually works
%              well)
% verbose    - Whether to print progess
%
% OUTPUT
% Vo - Output volume
%
% Patrick M. Jensen, 2019, Technical University of Denmark

% See source for mex file