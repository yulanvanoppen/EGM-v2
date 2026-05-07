function d = eucl_rel(x, y, ~)                                          % compute relative Euclidean distance
    if nargin < 3
        if iscell(x) && iscell(y)
            x = cell2mat(x);
            y = cell2mat(y);
        end
        x = flatten(x);
        y = flatten(y);
    end
    
    d = norm(x - y) / norm(x);

    if norm(x) == 0
        if norm(x - y) == 0
            d = 0;
        else
            d = Inf;
        end
    end
end