load('kmeansdata.mat')
[reduced_data]=PCA_data(X, 2)
scatter(X(:,1), X(:,2))
xlabel('pc1')
ylabel('pc2')
title('pca')