function [kmp] = K_elem_MP_6x6(AE,EI,L)
% Autor: Msc. Carlos Celi
% Calcula la matriz de rigidez de elemento tipo Marco Plano de tres
% acciones por nudo. {Axial,Corte,Momento}

% [kmp] = K_elem_MP_6x6(A,E,I,L)

% A = seccion transversal del elemento.
% E = Modulo de elasticidad del elemento.
% L = Longitud del elemento.
% I = Inercia del elemento.
kmp(:,1)=[AE/L;0;0;-AE/L;0;0];
kmp(:,2)=[0;12*EI/L^3;6*EI/L^2;0;-12*EI/L^3;6*EI/L^2];
kmp(:,3)=[0;6*EI/L^2;4*EI/L;0;-6*EI/L^2;2*EI/L];
kmp(:,4)=[-AE/L;0;0;AE/L;0;0];
kmp(:,5)=[0;-12*EI/L^3;-6*EI/L^2;0;12*EI/L^3;-6*EI/L^2];
kmp(:,6)=[0;6*EI/L^2;2*EI/L;0;-6*EI/L^2;4*EI/L];
end

