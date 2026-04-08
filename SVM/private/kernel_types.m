function K = kernel_types(type,input)
    switch lower(type)
        case 'poly'
            q = input{1};
            delta = input{2};
            K = @(X1,X2) polynomial_kernel(X1,X2,q,delta);
        case 'gaussian'
            sigma = input{1};
            K = @(X1,X2) gaussian_kernel(X1,X2,sigma);
        case 'sigmoid'
            kappa = input{1};
            delta = input{2};
            K = @(X1,X2) sigmoid_kernel(X1,X2,kappa,delta);
    end
end
function K = polynomial_kernel(X1,X2,q,delta)
    K = (X1*X2' + delta).^q;
end
function K = gaussian_kernel(X1,X2,sigma)
    n1 = size(X1,1); n2 = size(X2,1);
    dist = pdist2(X1,X2,'squaredeuclidean');
    K = exp(-dist ./ ones(n1,n2)*(2*sigma^2));
end
function K = sigmoid_kernel(X1,X2,kappa,delta)
    n1 = size(X1,1); n2 = size(X2,1);
    K = tanh(kappa.*X1*X2'- delta*ones(n1,n2));
end