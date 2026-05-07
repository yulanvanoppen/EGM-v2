function Ainv = tryinv(A)                                               % Wrapper combining testing and generalized inversion
    try
        Ainv = inv(A);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:nearlySingularMatrix')
            Ainv = pinv(A);
        end
    end
end