function C = tryNearestSPD(A)                                           % Wrapper combining testing SPD and correction
    try 
        B = chol(A);
        C = A;
    catch
        C = nearestSPD(A);
    end
end