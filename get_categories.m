function categories = get_categories (data, cutoffs)
% This function is for finding category for each feature.
% Cutoffs should be cell, in each cell should be saved the vector of
% cutoffs for each feature.

categories = zeros(length(cutoffs),1); % final variable 

for i = 1:length(cutoffs)
    for j = 1:length(cutoffs{i})
        if data(i)<= cutoffs{i}(j) % if value in data is smaller (or equal to) one of the cutoffs...
            categories(i) = j; % ...this position will be saved
        break;
        else 
            categories(i) = length(cutoffs{i})+1; % if the value is for last category
        end
    end
end