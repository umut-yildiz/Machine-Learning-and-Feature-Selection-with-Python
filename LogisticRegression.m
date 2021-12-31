function [Prediction] = LogisticRegression(training,dataset)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%       - Logistic Regression is one of the Supervised Learning Algorithms.
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
%       - Prediction: Nx1, Prediction based on Logistic Regression model.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = zeros(1,size(training,2));
prediction = 1/(1+exp((-(b(1)))));
alpha = 0.3;
X = ones(size(training,1),1);
training = [X,training];
y = training(:,end);

lastprediction = [];
for l = 1:10
    for h = 1:size(training,1)

        for j = 1:size(training,2)-1
            bnew(j) = b(j) + (alpha*(y(h)-prediction)*(prediction)*(1-prediction)*(training(h,j)));
        end
        b = [bnew];

        for i = 1:size(training,2)-1
            if h < size(training,1)
                a(1,i) = b(i)*training(h+1,i);
            end
        end

    newprediction = 1/(1+exp((-(sum(a)))));

    prediction = newprediction;
    end
end

% Prediction
B = [];
for i = 1:size(dataset,2)
       B = [B ,b(i+1).*dataset(:,i)];
end
B0 = repmat(b(1),size(dataset,1),1);
B = [B0,B];
LogModel = sum(B,2);
unlabeledprediction = 1./(1+exp((-(LogModel))));
Prediction = (unlabeledprediction>0.5);
end

