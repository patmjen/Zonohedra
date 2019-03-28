% NOTE: Make sure you have the correct CUDA installation and have set the 
% host compiler to something compatible with MATLAB's CUDA version.
% See here for CUDA: https://mathworks.com/help/parallel-computing/gpu-support-by-release.html;jsessionid=f52392b9e6a05b2fa8e3b910060a
% For a CUDA version less than 10.0 you likely need Visual Studio C++ 2015.
% See here for Visual Studio: https://devblogs.microsoft.com/cppblog/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017/
% Change host compiler with: mexcuda -setup C++

mexcuda mex_gorpho_flatLinearDilateErode3d_block.cu gorpho_matlab_utils.cu NVCC_FLAGS="-I../cudablockproc/lib -I../ --expt-relaxed-constexpr --use_fast_math"
mexcuda mex_gorpho_genDilateErode3d_block.cu gorpho_matlab_utils.cu NVCC_FLAGS="-I../cudablockproc/lib -I../ --expt-relaxed-constexpr --use_fast_math"
