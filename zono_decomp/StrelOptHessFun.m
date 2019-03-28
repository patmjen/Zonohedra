function H = StrelOptHessFun(A,Lambda,contain)
V1 = [0.5 0 1; 0.5 1 1; 0.5 2 1];
V2 = [0.5 0 0; 0.5 1 0; 0.5 2 2];
F1 = [0.5 2 2; 0 0 0; 0 0 0];
F2 = [0.5 1.5 1; 0.5 1.5 1; 0 0 0];
F3 = [0.5 1 1; 0.5,1,1; 0.5,1,1];
Ms = cat(3,V1,V2,F1,F2,F3);

H = zeros(3);
for i = 1:5
    M = Ms(:,:,i);
    Mx = M*A(1:3);
    NMx = sqrt(sum(Mx.^2,1) + eps);
    Gi = M'*(Mx)/NMx;
    Hi = (M'*M - Gi*Gi')/NMx;
    if contain == -1
        H = H + dot([1 1 -1], Lambda.ineqnonlin([0 5 10]+i))*Hi;
    elseif contain == 0
        H = H + dot([1 -1], Lambda.ineqnonlin([0 5]+i))*Hi;
    else
        H = H + dot([-1 1 -1], Lambda.ineqnonlin([0 5 10]+i))*Hi;
    end
end
H = [[H zeros(3,1)]; zeros(1,4)];