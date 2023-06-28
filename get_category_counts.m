function category_counts = get_category_counts(data, feature_index, cutoffs, label)
% This function counts in each category number of samples, which are
% labeles as 0 or as 1.
% For only 2 categories use one cutoff., e.g. for binary samples use
% cutoffs = [0];

% Extract the feature column
feature = data(:, feature_index);

% Calculate the counts of each category
category_counts = zeros(length(cutoffs)+1, 2); % final variable is bigger (+1) than number of cutoffs

% For loop for counting 
for i = 1:length(category_counts)
    if i == 1 % for first category
        category_counts(i, :) = [sum(feature <= cutoffs(i) & label == 1), ... % number of sample in category labeles as 1
                             sum(feature <= cutoffs(i) & label  == 0)]; % number of sample in category labeles as 0
    elseif i == length(category_counts) % for last category
                category_counts(i, :) = [sum(feature > cutoffs(i-1) & label == 1), ... % number of sample in category labeles as 1
                             sum(feature > cutoffs(i-1) & label  == 0)]; % number of sample in category labeles as 0
    else
        category_counts(i, :) = [sum(feature > cutoffs(i-1) & feature <= cutoffs(i) & label == 1),... % number of sample in category labeles as 1
                                  sum(feature > cutoffs(i-1) & feature <= cutoffs(i) & label == 0) ]; % number of sample in category labeles as 0
    end
    
end

end