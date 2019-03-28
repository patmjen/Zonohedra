function [A,d,success] = StrelSphereDecomp(r,contain,tolFrac)
% STRELSPHEREDECOMP Find zonohedral approximation of a sphere
% [A,d,success] = StrelSphereDecomp(r)
% [A,d,success] = StrelSphereDecomp(r,contain)
% [A,d,success] = StrelSphereDecomp(r,contain,tolFrac)
%
% INPUT
% r       - Radius of sphere to approximate
% contain - (optional, default: -1) Extra constraints for approximation:
%           * -1 = Approximation constrained to be *inside* sphere
%           *  0 = No extra constraint
%           *  1 = Approximation constrained to be *outside* sphere
% tolFrac - (optional: default: 1e-5) Tolerance for when a number is
%           considered to be an integer
%
% OUTPUT
% A       - Coefficients for approximation
% d       - Directed Haussdorff distance of approximation
% success - Did we find an approximation
%
% Patrick M. Jensen, 2019, Technical University of Denmark
if nargin < 2, contain = -1; end % default: contain inside sphere
if nargin < 3, tolFrac = 1e-5; end
LbCon = [1,0,0]';
UbCon = [inf,inf,inf]';

d = inf;
A = [];

ConFun = @(A) StrelOptConFun(A,r,contain);
numEval = 0;
FmcOpts = optimoptions('fmincon',...
    'Display','none',...
    'Algorithm','interior-point',...
    'SpecifyObjectiveGradient',true,...
    'SpecifyConstraintGradient',true,...
    'HessianFcn',@(A,Lambda) StrelOptHessFun(A,Lambda,contain),...
    'StepTolerance',1e-3);
[Ai,di,flagi] = SolveSubproblem(LbCon,UbCon);
SolveTree(LbCon,UbCon,Ai,di,flagi);

A = round(A);
success = ~isempty(A);

    function SolveTree(LbCon,UbCon,Ai,di,flagi)
        if flagi < 1
            % Could not solve so no feasible solution exists. Thus, this
            % subtree should be pruned
        elseif all(Frac(Ai,tolFrac) <= tolFrac)
            % Found integer solution so this subtree is done
            % We still prune if this solution is not an improvement
            if di < d
                d = di;
                A = Ai;
            end
        else
            % Found fractional solution. If an improvement is possible,
            % then branch
            if di < d
                % Branch on variable with largest fractional value
                [~,i] = max(Frac(Ai,tolFrac));
                
                UbConL = UbCon;
                UbConL(i) = floor(Ai(i));
                [AL,dL,flagL] = SolveSubproblem(LbCon,UbConL);
                if flagL < 1, dL = inf; end
                
                LbConR = LbCon;
                LbConR(i) = ceil(Ai(i));
                [AR,dR,flagR] = SolveSubproblem(LbConR,UbCon);
                if flagR < 1, dR = inf; end
                
                % Solve subtrees in order of best lower bound
                if dL < dR
                    SolveTree(LbCon,UbConL,AL,dL,flagL);
                    SolveTree(LbConR,UbCon,AR,dR,flagR);
                else
                    SolveTree(LbConR,UbCon,AR,dR,flagR);
                    SolveTree(LbCon,UbConL,AL,dL,flagL);
                end
            end
        end
    end

    function [Ai,di,flagi] = SolveSubproblem(LbCon,UbCon)
        numEval = numEval + 1;
        [Ahi,di,flagi] = fmincon(@(Ah) StrelOptObjFun(Ah),...
            [r/10 r/10 r/10 r]',[],[],[],[],[LbCon; -inf],[UbCon; inf],...
            ConFun,FmcOpts);
        Ai = Ahi(1:3);
    end
end

function f = Frac(x,tolFrac)
x = round(x/tolFrac)*tolFrac;
f = x - fix(x);
end