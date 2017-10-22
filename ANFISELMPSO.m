function [time, Et]=ANFISELMELM(adaptive_mode)
adaptive_mode = 'none';
tic;
clc;
clearvars -except adaptive_mode;
close all;

% load dataset
data = csvread('iris.csv');
input_data = data(:, 1:end-1);
output_data = data(:, end);

% Parameter initialization
[center,U] = fcm(input_data, 3, [2 100 1e-6]); %center = center cluster, U = membership level
[total_examples, total_features] = size(input_data);
class = 3; % [Changeable]
epoch = 0;
epochmax = 400; % [Changeable]
%Et = zeros(epochmax, 1);
[Yy, Ii] = max(U); % Yy = max value between both membership function, Ii = the class corresponding to the max value

% Population initialization
pop_size = 10;
population = zeros(pop_size, 3, class, total_features); % parameter: population size * 6 * total classes * total features
velocity = zeros(pop_size, 3, class, total_features); % velocity matrix of an iteration

c1 = 1.2;
c2 = 1.2;
original_c1 = c1;
original_c2 = c2;
r1 = 0.4;
r2 = 0.6;
max_c1c2 = 2;

% adaptive c1 c2
% adaptive_mode = 'none';
% class(adaptive_mode)
iteration_tolerance = 50;
iteration_counter = 0;
change_tolerance = 10;
is_first_on = 1;
is_trapped = 0;
%out_success = 0;
for particle=1:pop_size
    a = zeros(class, total_features);
    b = repmat(2, class, total_features);
    c = zeros(class, total_features);

    for k =1:class
        for i = 1:total_features % looping for all features
            % premise parameter: a
            aTemp = (max(input_data(:, i))-min(input_data(:, i)))/(2*sum(Ii' == k)-2);
            aLower = aTemp*0.5;
            aUpper = aTemp*1.5;
            a(k, i) = (aUpper-aLower).*rand()+aLower;

            %premise parameter: c
            dcc = (2.1-1.9).*rand()+1.9;
            cLower = center(k,total_features)-dcc/2;
            cUpper = center(k,total_features)+dcc/2;
            c(k,i) = (cUpper-cLower).*rand()+cLower;
        end
    end
    population(particle, 1, :, :) = a;
    population(particle, 2, :, :) = b;
    population(particle, 3, :, :) = c;
end

%inisialisasi pBest
pBest_fitness = repmat(100, pop_size, 1);
pBest_position = zeros(pop_size, 3, class, total_features);

% calculate fitness function
for i=1:pop_size
    particle_position = squeeze(population(i, :, :, :));
    e = get_fitness(particle_position, class, input_data, output_data);
    if e < pBest_fitness(i)
        pBest_fitness(i) = e;
        pBest_position(i, :, :, :) = particle_position;
    end
end

% find gBest
[gBest_fitness, idx] = min(pBest_fitness);
gBest_position = squeeze(pBest_position(idx, :, :, :));

% ITERATION
while epoch < epochmax
    epoch = epoch + 1;
    
    % calculate velocity and update particle
    % vi(t + 1) = wvi(t) + c1r1(pbi(t) - pi(t)) + c2r2(pg(t) - pi(t))
    % pi(t + 1) = pi(t) + vi(t + 1)
    r1 = rand();
    r2 = rand();
    for i=1:pop_size
        velocity(i, :, :, :) = squeeze(velocity(i, :, :, :)) + ((c1 * r1) .* (squeeze(pBest_position(i, :, :, :)) - squeeze(population(i, :, :, :)))) + ((c2 * r2) .* (gBest_position(:, :, :) - squeeze(population(i, :, :, :))));
        population(i, :, :, :) = population(i, :, :, :) + velocity(i ,:, :, :);
    end
    
    % calculate fitness value and update pBest
    for i=1:pop_size
        particle_position = squeeze(population(i, :, :, :));
        e = get_fitness(particle_position, class, input_data, output_data);
        if e < pBest_fitness(i)
            pBest_fitness(i) = e;
            pBest_position(i, :, :, :) = particle_position;
        end
    end
    
    % find gBest
    [gBest_fitness, idx] = min(pBest_fitness);
    gBest_position = squeeze(pBest_position(idx, :, :, :));
    
    Et(epoch) = gBest_fitness;
    
    % Adaptive c1 and c2
    if strcmpi('none', adaptive_mode) == 0
        if (epoch > 1) && (Et(epoch) == Et(epoch-1))
            iteration_counter = iteration_counter + 1;
        else
            iteration_counter = 0;
            c1 = original_c1;
            c2 = original_c2;
%             if is_trapped == 1
%                 out_success = out_success + 1;
%                 is_trapped = 0;
%             end
        end

        if iteration_counter > iteration_tolerance
%             is_trapped = 1;
            if is_first_on == 1
                iteration_left_since_on = (epochmax - epoch) - change_tolerance;
                epoch_when_on = epoch;
                is_first_on = 0;
            end
            curr_pos = (epoch-epoch_when_on);
            if curr_pos <= iteration_left_since_on
                if strcmpi('c1', adaptive_mode) == 1
                    c1 = original_c1 + (max_c1c2 - original_c1)/iteration_left_since_on*(curr_pos);
                    %fprintf('iter %d - %d\n', epoch, c1)
                elseif strcmpi('c2', adaptive_mode) == 1
                    c2 = original_c2 + (max_c1c2 - original_c2)/iteration_left_since_on*(curr_pos);
                    %fprintf('iter %d - %d\n', epoch, c2)
                elseif strcmpi('c1c2', adaptive_mode) == 1
                    c1 = original_c1 + (max_c1c2 - original_c1)/iteration_left_since_on*(curr_pos);
                    c2 = original_c2 + (max_c1c2 - original_c2)/iteration_left_since_on*(curr_pos);
                end
            end
        end
    end
    
    % Draw the SSE plot
%     plot(1:epoch, Et);
%     title(['Epoch  ' int2str(epoch) ' -> MSE = ' num2str(Et(epoch))]);
%     grid
%     pause(0.001);
end

%[out output out-output]
% ----------------------------------------------------------------
time = toc;
end