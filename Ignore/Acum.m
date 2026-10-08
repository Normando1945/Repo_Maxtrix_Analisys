function [P] = Acum(Lee,aep_L,P)
% Autor: Msc. Carlos Celi
% Acumula las acciones de nudo correspondientes al mismo GDL

% [PT] = Acum(Lee,aep_L,P)

% Lee   = Vector de colocación del elemento.
% aep_L = Vector de acciones de nudo (empotramiento perfecto), de elemento, en coordenadas locales.
% P     = Vector de acciones de nudo en coordenadas globales.
for i=1:length(Lee)                                       
    ii = Lee(i);                                    
    P(ii)=P(ii)- aep_L(i);
end

end

