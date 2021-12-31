function [Prediction] = OutlierDetection(dataset)
[nrow,ncolumn] = size(dataset);
test = dataset;
test = reshape(test,1,nrow.*ncolumn);
test = transpose(test);
[result] = feat_sel_f.NormalityTest(test); % Normality Test

if result==1
    Outlier = feat_sel_f.GrubbsOutlierDetection(dataset); % Grubbs method for normal

else
    Outlier = feat_sel_f.QuartilesOutlierDetection(dataset); % Quartiles method 
    
end

Observation = sum(Outlier,2);
Prediction = Observation>0;
end

