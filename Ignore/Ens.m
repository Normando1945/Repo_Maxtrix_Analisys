function [S] = Ens(lee,K,S,nglt)
% Ens vector de ensamble %
% lee vector de colocacion. numerando los grados de libertad para cada elemento %
% K Matrices de Rigides de cada elemento en coordenadas de globales%
% S Matriz de rigidez total de la estructura %
% nglt numero de grados de libertad incluido resticciones %
ng = length(lee);
for i=1:ng                                          % contador de filas %
    ii = lee(i);                                    % cuanta la posiscion ii en 1,1:2,2..etc%
    if ii>0,
        if ii <= nglt,                              % restringe que que las posicio ii no supere los grados de libertad%
            for j=1:ng,
                jj=lee(j);                          % cuanta la posiscion jj en 1,1:2,2..etc%
                if jj>0,
                    if jj <= nglt,                  % restringe que que las posicio jj no supere los grados de libertad%
                        S(ii,jj)=S(ii,jj)+ K(i,j);  % ensambla la matriz S con las coordenadas mas la matriz original K de cada miembro%
                    end
                end
            end
        end
    end
end
end
