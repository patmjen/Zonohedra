function [C,Ceq,GC,GCeq] = StrelOptConFun(Ah,r,contain)
Ceq = [];
GCeq = [];
[D,GD] = StrelZonoSphereDist(Ah(1:3),r);
if contain == -1
    C = [D, D - Ah(4), -D - Ah(4)]; 
elseif contain == 0
    C = [D - Ah(4), -D - Ah(4)]; 
else
    C = [-D, D - Ah(4), -D - Ah(4)]; 
end
if nargout > 2
    GC1 = [GD; zeros(1,size(GD,2))];
    GC2 = [GD; -ones(1,size(GD,2))];
    GC3 = [-GD; -ones(1,size(GD,2))];
    if contain == -1
        GC = [GC1 GC2 GC3];
    elseif contain == 0
        GC = [GC2 GC3];
    else
        GC = [-GC1 GC2 GC3];
    end
end