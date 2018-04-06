function [f] = acopB(w,x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here



gB=4e-14;%4x10-11 m/w,ganancia brillouin
Aeff=85e-18;%50e-18;%area efectiva

a=0.2/4.343;%atenuacion alfa 2dB/KM
b=gB/(1.2*Aeff);%b=gB/(1.2*Aeff);

f(1)=-a*x(1)-b*x(1)*x(2);
f(2)=a*x(2)-b*x(1)*x(2);
f=f';
end



