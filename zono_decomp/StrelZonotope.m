function G = StrelZonotope(a1,a2,a3)
% STRELZONOTOPE Structure element zonotope
% G = StrelZonotope(a1,a2,a3)
%
% INPUT
% a1 - Axes
% a2 - Plane diagonals
% a3 - Octant diagonals
%
% OUTPUT
% G - 3 x N array with zonotope generating vectors as columns
%
% Patrick M. Jensen, 2018

G = zeros(3,0);
if a1 > 0
    G = [G eye(3)*a1];
end
if a2 > 0
    G = [G [1 1 0; 1 -1 0; 1 0 1; 1 0 -1; 0 1 1; 0 1 -1]'*a2];
end
if a3 > 0
    G = [G [1 1 1; 1 1 -1; 1 -1 1; -1 1 1]'*a3];
end