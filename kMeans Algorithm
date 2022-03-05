%% K (K=2) Means Algorithm
function [Prediction] = kMeans(dataset)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%       - kMeans is one of the Unsupervised Learning Algorithms.
%       - This function separates samples in two clusters.
%
% Input: 
%       - dataset: NxM (N:represent samples(observations), M:represent
%       features).
%
% Output: 
%       - Prediction: Nx1, 0 represent good samples, 1 represent bad 
%       samples.
%
% Note: 
%       - This function is for only 2 classes (k=2).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Define the center points for each cluster (randomly)
center1 = dataset(1,:); 
center2 = dataset(2,:);

cluster1 = [];
cluster2 = [];
% clusters are created by associating every observation with the nearest 
% mean
for i = 1:size(dataset,1)
    cluster1 = [cluster1; sum(abs(dataset(i,:) - center1))]; 
    cluster2 = [cluster2; sum(abs(dataset(i,:) - center2))];
end

label = (cluster1 < cluster2); %split clusters

newdata = [dataset, label];

%The centroid of each of the k clusters becomes the new mean.
center1 = mean(newdata(label==1,:));
center2 = mean(newdata(label==0,:));

%Remove the labels
center1(:,end) = [];
center2(:,end) = [];

%Compute clusters again
cluster1 = [];
cluster2 = [];
for i = 1:size(dataset,1)
    cluster1 = [cluster1; sum(abs(dataset(i,:) - center1))];
    cluster2 = [cluster2; sum(abs(dataset(i,:) - center2))];
end

Prediction = [cluster1<cluster2]; %split clusters
IterationStop = label==Prediction; %condition of iteration

%do all steps until when last result would same before the last result
while any(IterationStop(:,1) ~= 1) 
    
cluster1 = [];
cluster2 = [];

for i = 1:size(dataset,1)
    cluster1 = [cluster1; sum(abs(dataset(i,:) - center1))];
    cluster2 = [cluster2; sum(abs(dataset(i,:) - center2))];
end

label = (cluster1 < cluster2);

cluster = [cluster1, cluster2, label];
newdata = [dataset, label];

center1 = mean(newdata(label==1,:));
center2 = mean(newdata(label==0,:));
center1(:,end) = [];
center2(:,end) = [];


cluster1 = [];
cluster2 = [];

for i = 1:size(dataset,1)
    cluster1 = [cluster1; sum(abs(dataset(i,:) - center1))];
    cluster2 = [cluster2; sum(abs(dataset(i,:) - center2))];
end

label = (cluster1 < cluster2);

cluster = [cluster1, cluster2, label];
newdata = [dataset, label];

center1 = mean(newdata(label==1,:));
center2 = mean(newdata(label==0,:));
center1(:,end) = [];
center2(:,end) = [];

cluster1 = [];
cluster2 = [];
for i = 1:size(dataset,1)
    cluster1 = [cluster1; sum(abs(dataset(i,:) - center1))];
    cluster2 = [cluster2; sum(abs(dataset(i,:) - center2))];
end

Prediction = [cluster1<cluster2];
IterationStop = label==Prediction;

end

if mean(cluster1)>mean(cluster2)
    Prediction = [cluster1>cluster2];
else
    Prediction = [cluster1<cluster2];
end

end
