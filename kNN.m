%% kNN Algorithm
function [Prediction] = kNN(dataset,test)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%       - kNN is one of the Supervised Learning Algorithms.
%       - This function classifies samples in two class.
%
% Input:
%       - training: Dataset that wanted to train
%       NxM (N:represent samples(observations), M:represent
%       features).
%       - target: Nx1, label, should be 0,1,2,... format.
%       - test: (NxM (N:represent samples(observations), M:represent
%       features).
%
% Output:
%       - Prediction: Nx1, Prediction based on kNN.
%
% Note: 
%       - Function is calculate distance that euclidean.
%       - This function is edited for two class, otherwise k should be
%       changed (line 24).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
k = 2;
training = dataset(:,1:end-1);
target = dataset(:,end);

%initialization
Prediction=zeros(size(test,1),1);
ed=zeros(size(test,1),size(training,1)); %ed: (MxN) euclidean distances 
ind=zeros(size(test,1),size(training,1)); %corresponding indices (MxN)
k_nn=zeros(size(test,1),k); %k-nearest neighbors for testing sample (Mxk)

%calc euclidean distances between each testing data point and the training
%data samples
for test_point=1:size(test,1)
    for train_point=1:size(training,1)
        %calc and store sorted euclidean distances with corresponding indices
        ed(test_point,train_point)=sqrt(...
            sum((test(test_point,:)-training(train_point,:)).^2));
    end
    [ed(test_point,:),ind(test_point,:)]=sort(ed(test_point,:));
end

%find the nearest k for each data point of the testing data
k_nn=ind(:,1:k);
%get the majority vote 
for i=1:size(k_nn,1)
    options=unique(target(k_nn(i,:)'));
    max_count=0;
    max_label=0;
    for j=1:length(options)
        L=length(find(target(k_nn(i,:)')==options(j)));
        if L>max_count
            max_label=options(j);
            max_count=L;
        end
    end
    Prediction(i)=max_label;
end
Prediction = Prediction == 1;
end
