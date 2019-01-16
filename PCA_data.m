%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [reduced_data]=PCA_data(input_data, num_components);
n = size(input_data, 1);
b = sum(input_data, 1)./ n; 
C = (1/n) * ((input_data - b)' * (input_data-b));
[eigen_vec, eigen_val] = eigs(C, num_components);  % get the num_components biggest eigenvalues
[d, index] = sort(diag(eigen_val), 'descend'); % sort the eigenvalues
eigen_val = eigen_val(index, index);
eigen_vec = eigen_vec(:, index);
W = eigen_vec;
reduced_data = (input_data-b) * W;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
