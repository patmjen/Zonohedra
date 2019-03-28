function [D,GD] = StrelZonoSphereDist(A,r)
if isrow(A), A = A'; end

V1 = [0.5 0 1; 0.5 1 1; 0.5 2 1];
V2 = [0.5 0 0; 0.5 1 0; 0.5 2 2];
F1 = [0.5 2 2; 0 0 0; 0 0 0];
F2 = [0.5 1.5 1; 0.5 1.5 1; 0 0 0];
F3 = [0.5 1 1; 0.5,1,1; 0.5,1,1];
% F4 = [0.5,2,1; 0.5,0.5,1; 0.5,0.5,1];

% d1 = [0,1,-1]';
% d2 = [-1,1,1]';
% d3 = [2,1,1]';
% M = [d1/norm(d1),d2/norm(d2),d3/norm(d3)];
% f4 = F4*A;
% f4 = M'*f4;
% f4(2) = min(f4(2),0);
% f4 = M*f4;

% Pts = [V1*A V2*A F1*A F2*A F3*A f4];
Pts = [V1*A V2*A F1*A F2*A F3*A];
NPts = sqrt(sum(Pts.^2,1) + eps);
D = NPts - r;
% GD = zeros(3,6);
GD = zeros(3,5);
GD(:,1) = V1'*(V1*A)/NPts(1);
GD(:,2) = V2'*(V2*A)/NPts(2);
GD(:,3) = F1'*(F1*A)/NPts(3);
GD(:,4) = F2'*(F2*A)/NPts(4);
GD(:,5) = F3'*(F3*A)/NPts(5);
% GD(:,6) = F4'*(F4*A)/NPts(6);
    