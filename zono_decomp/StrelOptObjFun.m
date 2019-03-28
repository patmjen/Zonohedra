function [d,Gd] = StrelOptObjFun(Ah)
d = Ah(4);
Gd = [0 0 0 1];