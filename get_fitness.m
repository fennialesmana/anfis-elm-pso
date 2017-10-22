function fitness = get_fitness(premise, class, input_data, output_data)
%{
get_fitness = Calculate fitness value for each premise parameter (a, b, c) of one particle

*) Output Parameter:
    - fitness = fitness value result
*) Input Parameter:
    - premise = premise parameter, size: 3 (a, b, and c parameter) x total
    classes x total features
    - class   = total classes
    - data    = input data, total samples x total features
%}

[total_examples, total_features] = size(input_data);
H = [];
Mu = zeros(total_examples, class, total_features); % Mu: miu all samples (total samples x total classes x total features)

% forward pass
for i = 1:total_examples
    for k = 1:class
        w1(k) = 1; % w (not w bar)
        for j = 1:total_features
            mu(k, j) = 1/(1 + ((input_data(i, j)-premise(3, k, j))/premise(1, k, j))^(2*premise(2, k, j))); % mu: miu of one sample
            w1(k) = w1(k)*mu(k, j); % fill w of k-th class
            Mu(i, k, j) = mu(k, j);
        end
    end
    w = w1/sum(w1); % w = w bar of one row / one sample data
    XX = [];
    for k = 1:class
        XX = [XX w(k)*input_data(i,:) w(k)];
    end % in the end, XX is H matrix for one sample in coresponding iteration
    H = [H; XX]; % combine matrix H of each sample
end
% end of forward pass

% find consequent parameter (p, q, r)
beta = pinv(H) * output_data; % moore pseudo invers
weight = H * beta; % calculate weight of hidden to output node
fitness = sum((output_data - weight).^2)/total_examples; % calculate error using MSE

end