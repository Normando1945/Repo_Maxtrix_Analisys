function [Ta] = T_elem_MP_6x6(alfas)
% Autor: Msc. Carlos Celi
% Permite pasar de un sistema de coordenadas locales a globales.

% T_elem_MP_6x6(alfas)

% alfas = angulo comprendido entre el sistema de coordenadas globales hacia
% el sistema de coordenadas locales
Ta=[cos(alfas*pi/180) -sin(alfas*pi/180) 0 0 0 0
    sin(alfas*pi/180) cos(alfas*pi/180) 0 0 0 0
    0 0 1 0 0 0
    0 0 0 cos(alfas*pi/180) -sin(alfas*pi/180) 0
    0 0 0 sin(alfas*pi/180) cos(alfas*pi/180) 0
    0 0 0 0 0 1];
end

