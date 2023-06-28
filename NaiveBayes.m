% AI in healthcare
% Lab Assignment - Naive Bayes
% Author: Klara Martinaskova
%
% Task:
% The task is to train Naive Bayes classifier based on training dataset 
% and estimate the cardiovascular disease occurence of the subjects 
% in testing dataset.
%
% I decided to make a MATLAB script even though I think excel is easier, 
% but MATLAB is more challenging. And so I think it is better to apply 
% MATLAB script (this model) to different data with different sample sizes.

close all
clear all
clc
%% Loading data
train = load('Naive Bayes classifier task - training group.txt');
test = load('Naive Bayes classifier task - test group.txt');

features = {'Gender','Age','Height','Weight','BMI','Workout per week',...
    'Work attribute','Smoking status','Radial artery wall thickness',...
    'Femoral artery wall thickness','Apo-B','Carotid artery wall thickness',...
    'Aortic pulse wave velocity','Total Cholesterol','LDL Cholesterol', ...
    'Systolic blood pressure','Diastolic blood pressure'}; % names of columns 

% last column is final diagnosis in train dataset
label = train(:,end); % labels
train = train(:,1:end-1); % features 
%% Correlation with results
corr_coeffs = zeros(1, size(train, 2)); % helper var. for saving results

% For loop for finding correlations coeff. with results (diagnosis)
for i = 1:size(train, 2) 
    corr_coeffs(i) = abs(corr(train(:, i), label)); % calculation of corr (negative corr is transfer to positive values with abs)
end

% Plot results of correlations
figure(1)
bar(corr_coeffs);
xticklabels(features);
xticks(1:size(train, 2));
xtickangle(90);
ylabel('Correlation Coefficient');
yline(0.5, '--', 'LineWidth', 2, 'Color', 'r'); % add line to corr = 0.5
yline(0.3, ':', 'LineWidth', 2, 'Color', 'r'); % add line to corr = 0.3
title('Correlation - features and final diagnosis')

%disp(features(corr_coeffs>0.5))
%disp(features(corr_coeffs>0.3))

%% Correlation with results - choosing features
% I have chosen the features, which have higher corr. coef. with result higher than
% 0.3 (this is demonstrated in the graph).
% Chosen features:
% Higher than 0.5:
% 9, 11, 14, 16
%  'Radial artery wall thickness', 'Apo-B','Total Cholesterol','Systolic blood pressure'
% Higher than 0.3:
% 'Smoking status', 'Radial artery wall thickness', 'Apo-B', 'Aortic pulse wave velocity',...
% 'Total Cholesterol','LDL Cholesterol', 'Systolic blood pressure','Diastolic blood pressure'
%% Conditioning of the features 
% Choosing only independent features.
% Calculate the correlation matrix
corr_matrix = corr(train); % count correlation matrix

% Setting a correlation threshold 
correlation_threshold = 0.7; % 0.7 has been chosen according to dataset

% Find highly correlated features
highly_correlated_features = []; % variable for saving features with high correlation
for i = 1:size(corr_matrix, 1) % go over the correlation matrix
    for j = i+1:size(corr_matrix, 2)
        if abs(corr_matrix(i,j)) >= correlation_threshold % compare with threshold
            highly_correlated_features = [highly_correlated_features, j]; % saving features with high correlation
        end
    end
end

figure(2)
imagesc(corr_matrix);
title('Correlation between each two features')
xticklabels(features);
xticks(1:size(train, 2));
xtickangle(90);
yticklabels(features);
yticks(1:size(train, 2));
colorbar;

% Remove highly correlated features
train(:, unique(highly_correlated_features)) = []; % unique in case with repeating same positions
features(unique(highly_correlated_features)) = [];
%% Correlation with results again, but without removed features
% Choosing only independent features.

corr_coeffs = zeros(1, size(train, 2)); % helper var. for saving results of correlation

% For loop for finding correlations coeff. with results (diagnosis)
for i = 1:size(train, 2)
    corr_coeffs(i) = abs(corr(train(:, i), label)); % calculation of corr (negative corr is transfer to positive values with abs)
end

% Plot results of correlations
figure(3)
bar(corr_coeffs);
xticklabels(features);
xticks(1:size(train, 2));
xtickangle(90);
ylabel('Correlation Coefficient');
yline(0.5, '--', 'LineWidth', 2, 'Color', 'r'); % add line to corr = 0.5
yline(0.3, ':', 'LineWidth', 2, 'Color', 'r'); % add line to corr = 0.3
title('Correlation - independent features and final diagnosis')

%disp(features(corr_coeffs>0.5))
%disp(features(corr_coeffs>0.3))

% I have chosen the features, which have higher corr. coeff. with result higher than
% 0.3 (this is demonstrated in the graph) and are independent.
% Chosen features:
% Higher than 0.5:
% 'Radial artery wall thickness', 'Apo-B','Systolic blood pressure' 
% Higher than 0.3:
% 'Smoking status', 'Radial artery wall thickness', 'Apo-B','LDL Cholesterol','Systolic blood pressure'
%% Discretization and feature modelling
% For the sake of clarity, I use indexing from the original train file.
selected_features = find(corr_coeffs>0.3);
% 5     6     8    10    11 = 'Smoking status', 'Radial artery wall thickness', 'Apo-B','LDL Cholesterol','Systolic blood pressure'
% train = train(:,selected_features); 

%% Smoking
% Smoking status is binary, so no modelling is necessary

% Cutoff - need just 2 categories
smoking_cutoffs = [0]; % because there are only binary samples, use 0 as a cutoff
smoking_counts = get_category_counts(train, 5, smoking_cutoffs, label); % 5 = 'Smoking status' has index 8

%%  'Radial artery wall thickness'
% There is continuous data, it is necessary to categorize it. 
% The normal range of radial artery wall thickness in healthy individuals is considered to be between 0.2 and 0.5 mm.
% Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2734005/

% Create two categories  - higher than 0.5 and smaller  than 0.5 (or equal to 0.5)
RA_wall_thick_cutoffs = [0.5];
RA_wall_thick_counts = get_category_counts(train, 6, RA_wall_thick_cutoffs, label); % 6 = for index 'Radial artery wall thickness'

%% 'Apo-B'
% APO-B (Apolipoprotein B) is a protein that is found in the outer membrane
% of low-density lipoprotein (LDL) particles, which are often referred to 
% as "bad cholesterol." 
% Sources:
% https://www.urmc.rochester.edu/encyclopedia/content.aspx?contenttypeid=167&contentid=apolipoprotein_b100
% https://emedicine.medscape.com/article/2087335-overview
% https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8540246/
% "the mean apoB concentration was 1.31 ± 0.35 g/L in males versus 1.22 ±
% 0.36 g/L in females"
%
% I decided to create categories:
% x =< 1.66
% 1.66 < x < 2
% 2 < x < 3
% x =< 3

% Define the cutoffs for the APO-B categories
apoB_cutoffs = [1.66, 2, 3];
apoB_counts = get_category_counts(train, 8, apoB_cutoffs, label); % 8 = apoB has index 8

%% 'LDL Cholesterol'
% Source: https://my.clevelandclinic.org/health/articles/24391-ldl-cholesterol
% I decided create this cutoffs:
% Normal: Below 100 mg/dL.
% Near optimal: 100 – 129 mg/dL.
% Borderline high: 130 – 159 mg/dL.
% High: 160 – 189 mg/dL.
% Very high: 190 mg/dL or higher.

% Define the cutoffs for the LDL cholesterol categories
ldl_cutoffs = [1, 1.29, 1.6, 1.89];
ldl_counts = get_category_counts(train, 10, ldl_cutoffs, label); % 10 = 'LDL cholesterol' has index 10

%% 'Systolic blood pressure'
% Sources:
% https://www.cdc.gov/bloodpressure/about.htm
% https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.121.054602
%
% Normal: less than 120 mm Hg
% Elevated: 120-129 mm Hg
% Stage 1 hypertension: 130-139 mm Hg
% Stage 2 hypertension: 140 mm Hg or higher

% Define the cutoffs for the systolic blood pressure categories
sbp_cutoffs = [120, 129, 139];
sbp_counts = get_category_counts(train, 11, sbp_cutoffs, label); % 11 = 'Systolic blood pressure' has index 11

%% Naive Bayes classifier
% Create table of probabilities

% Total number of patients labeled as 1 - withdisease
total_1 = sum(label == 1);
% Total number of patients labeled as 0 - without disease
total_0 = sum(label == 0);

% Put into one cell
train_sum = { smoking_counts;
                RA_wall_thick_counts;
                apoB_counts;
                ldl_counts;
                sbp_counts};

% For counting conditional probability P(x|c) - use cell with same structures as train_sum            
probabilities_cell = train_sum; % for saving P(x|c), same structure as train_sum

% Two for loops for counting P(x|c)
for i = 1:length(train_sum)
    for j = 1 : length(train_sum{i})
        probabilities_cell{i}(j,1) = train_sum{i}(j,1)/total_1; % for labeled as 1
        probabilities_cell{i}(j,2) = train_sum{i}(j,2)/total_0; % for labeled as 0
    end
end

P_c1 = total_1/(total_1+total_0); % probabilty of "diagnosis of disease" labeled as 1 - with disease
P_c0 = total_0/(total_1+total_0); % probabilty of "diagnosis of disease" labeled as 0 - without disease 
P_c = [P_c1; P_c0];
%% Estimation of cardiovascular disease occurence of the subjects in testing dataset
% I will use same condition for data categorization as in the training set.

% Remove dependent features
test(:, unique(highly_correlated_features)) = [];

% Remindig cutoffs
% smoking_cutoffs = [0]; 
% RA_wall_thick_cutoffs = [0.5];
% apoB_cutoffs = [1.66, 2, 3];
% ldl_cutoffs = [1, 1.29, 1.6, 1.89];
% sbp_cutoffs = [120, 129, 139];

% Puttung all cutoffs to one cell
cutoffs_cell = {smoking_cutoffs; 
RA_wall_thick_cutoffs ;
apoB_cutoffs;
ldl_cutoffs;
sbp_cutoffs};

% Choose only selected features from testing data set
test_features = test(:,selected_features); 

[samples,~] = size(test_features); % number of patient
test_predict = zeros(samples,1); % for final results of classification

% For each patient find categories thanks cutoffs and labeled them as 0 or 1
for i = 1:samples
    categories = get_categories (test_features(i,:), cutoffs_cell); % find categories 
    use_P = zeros(length(categories),2); % for saving useful probabilites from probabilities_cell
    for j = 1:length(categories)
        use_P(j,:) = probabilities_cell{j}(categories(j),:); % save probabilities according to categories      
    end   
    
    % Save final probabilities for labeling as 1 or as 0 
    C_1 = prod(use_P(:,1))*P_c1; % prod returns the product of the array elements
    C_0 = prod(use_P(:,2))*P_c0;
    
    % The prediction of labeling
    if C_1 > C_0 % comparing counting samples
        test_predict(i) = 1; % cardiovascular disease
    else
        test_predict(i) = 0; % without cardiovascular disease
    end
end

%% Printing final results
disp('Predicted classification of patients into groups:')
disp(test_predict)