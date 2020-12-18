%Read in data
T = readtable('FinalAllSeasonData2.csv','Delimiter',',');
[n,~] = size(T);

%Randomly assign training and testing sets
n70 = round(0.70 * n);
rand70 = randperm(n, n70);
T_Train = T(rand70,:);
T_Test = T;
T_Test(rand70,:) = [];

%% Naive Model

pred_naive = rand(round(0.30*n),1);
pred_naive(pred_naive >= 0.39) = 1;
pred_naive(pred_naive < 0.39) = 0;

pred_naive = ones(round(0.3*n),1);
accuracyNAIVE1 = 1-sum(abs(pred_naive-T_Test.PlayType))/(n*0.3)                                 %Calculate accuracy

%% Cart Model

CART = fitctree(T_Train, 'PlayType','MaxNumSplits', 251);       %Create CART model
%view(tree.Trained{1}, 'Mode', 'graph')                                              %View one instance of a tree
pred_CART = predict(CART, T_Test);                                            %Predict on testing set
accuracyCART = 1-sum(abs(pred_CART-T_Test.PlayType))/(n*0.3)                                 %Calculate accuracy

%% Random Forest Model

forest = TreeBagger(75, T_Train, 'PlayType');                                       %Create Random Forest model  
pred_forest = str2double(predict(forest, T_Test));                                  %Predict on testing set  
accuracyFOREST = 1-sum(abs(pred_forest-T_Test.PlayType))/(n*0.3)                          %Calculate accuracy

%% Logistic Regression Model

% delete column for yardline as it isn't statistically significant
T_Train_log = T_Train(:,[1 2 3 4 5 7 8]);  
T_Test_log = T_Test(:,[1 2 3 4 5 7 8]);

logistic = fitglm(T_Train_log, 'Distribution','binomial', 'Link', 'logit');
disp(logistic);

pred_logistic = predict(logistic, T_Test_log);
pred_logistic(pred_logistic < 0.5) = 0;
pred_logistic(pred_logistic >= 0.5) = 1;

accuracyLOGISTIC = 1-sum(abs(pred_logistic-T_Test.PlayType))/(n*0.3)

%[X,Y,T,AUC] = perfcurve(T_Test.PlayType, pred_logistic, 1);
% plot(X,Y)
% ylabel('True Positive Rate')
% xlabel('False Positive Rate')

%% AdaBoosting Model

learn = [1 0.1 0.01];
accuracies2 = zeros(3,50);
k = 1;

for ii = 3:3
    for jj = 1:20:1000
        ada_boost = fitensemble(T_Train, 'PlayType', 'AdaBoostM1', 500, 'Tree','type','classification', 'LearnRate', 1);
        pred_ada = predict(ada_boost, T_Test);
        accuracyADABOOST = 1-sum(abs(pred_ada-T_Test.PlayType))/(n*0.3)
        
        accuracies2(ii,k) = accuracyADABOOST;
        k=k+1
    end
end


%% Gradient Boosting Model

gradient_boost = fitensemble(T_Train, 'PlayType', 'LSBoost', 500, 'Tree', 'LearnRate', 1);
pred_gradient = predict(gradient_boost, T_Test);
accuracyGRADIENT = 1-sum(abs(pred_gradient-T_Test.PlayType))/(n*0.3)

%% K Nearest Neighbours

k = 1;

for ii = 10:30:360

    knn = fitcknn(T_Train, 'PlayType','Distance', 'minkowski', 'DistanceWeight', 'inverse', 'NumNeighbors', ii);
    pred_knn = predict(knn, T_Test);
    accuracyKNN = 1-sum(abs(pred_knn-T_Test.PlayType))/(n*0.3);
    
    knnAccuracy(5,k) = accuracyKNN;
    k=k+1
    
end

%% Cross Validation Testing

T = readtable('FinalAllSeasonData.csv','Delimiter',',');
[n,~] = size(T);

tic
accuracyNAIVE1 = zeros(10,1);
accuracyNAIVE2 = zeros(10,1);
accuracyCART = zeros(10,1);
accuracyFOREST = zeros(10,1);
accuracyADA = zeros(10,1);
accuracyGRADIENT = zeros(10,1);
accuracyLOGISTIC = zeros(10,1);
accuracyKNN = zeros(10,1);

for ii = 1:3
    CVO = cvpartition(T.PlayType,'KFold',10);
    for jj = 1:CVO.NumTestSets     
        
        if jj == 2
            break
        end
        
        trainIndex = CVO.training(jj);
        testIndex = CVO.test(jj);
        test_actual = T(testIndex,8).PlayType;
        
        %Naive Version 1
        pred_naive1 = rand(CVO.TestSize(jj),1);
        pred_naive1(pred_naive1 >= 0.39) = 1;
        pred_naive1(pred_naive1 < 0.39) = 0;
        accuracyNAIVE1(10*(ii-1)+jj) = (1-sum(abs(pred_naive1-test_actual)/CVO.TestSize(jj)));
        
        %Naive Version 2
        pred_naive2 = ones(CVO.TestSize(jj),1);
        accuracyNAIVE2(10*(ii-1)+jj) = (1-sum(abs(pred_naive2-test_actual)/CVO.TestSize(jj)));

        %CART
        CART = fitctree(T(trainIndex,:), 'PlayType','MaxNumSplits', 251);       
        pred_CART = predict(CART, T(testIndex,:));                                            
        accuracyCART(10*(ii-1)+jj) = (1-sum(abs(pred_CART-test_actual)/CVO.TestSize(jj)));      
        
        %Random Forest
        forest = TreeBagger(75, T(trainIndex,:), 'PlayType');                                        
        pred_forest = str2double(predict(forest,  T(testIndex,:)));                                
        accuracyFOREST(10*(ii-1)+jj) = 1-sum(abs(pred_forest-test_actual)/CVO.TestSize(jj)); 
        
        %AdaBoost
        ada_boost = fitensemble(T(trainIndex,:), 'PlayType', 'AdaBoostM1', 250, 'Tree', 'LearnRate', 1);
        pred_ada = predict(ada_boost, T(testIndex,:));
        accuracyADA(10*(ii-1)+jj) =  (1-sum(abs(pred_ada-test_actual)/CVO.TestSize(jj)));   
        
        %Gradient Boosting
        gradient_boost = fitensemble(T(trainIndex,:), 'PlayType', 'LSBoost', 250, 'Tree', 'LearnRate', 1);
        pred_gradient = predict(gradient_boost, T(testIndex,:));
        accuracyGRADIENT(10*(ii-1)+jj) = (1-sum(abs(pred_gradient-test_actual)/CVO.TestSize(jj)));  
        
        %Logistic
        logistic = fitglm(T(trainIndex,[1 2 3 4 5 7 8]), 'Distribution','binomial', 'Link', 'logit');
        pred_logistic = predict(logistic, T(testIndex,[1 2 3 4 5 7 8]));
        pred_logistic(pred_logistic < 0.5) = 0;
        pred_logistic(pred_logistic >= 0.5) = 1;
        accuracyLOGISTIC(10*(ii-1)+jj) = (1-sum(abs(pred_logistic-test_actual)/CVO.TestSize(jj)));
        
        %K-Nearest Neighbours
        knn = fitcknn(T(trainIndex,:), 'PlayType','Distance', 'seuclidean', 'DistanceWeight', 'squaredinverse', 'NumNeighbors', 250);
        pred_knn = predict(knn, T(testIndex,:));
        accuracyKNN(10*(ii-1)+jj) =  (1-sum(abs(pred_knn-test_actual)/CVO.TestSize(jj)));
    end
end
save('data2', 'accuracyNAIVE1','accuracyNAIVE2','accuracyCART','accuracyFOREST','accuracyADA','accuracyGRADIENT','accuracyLOGISTIC', 'accuracyKNN');
time = toc;

%% T Test

% figure
% boxplot( [accuracyNAIVE2(:), accuracyADA(:)],'color', 'rb','Labels',{'Naive','AdaBoost'}, 'Widths', 1);
% grid
% title('Comparison of Prediction Accuracy'); 
% ylabel('Prediction Accuracy');
% xlabel('Model');
% set(gca,'xtick');
% set(gca,'xticklabel',{'Naive','AdaBoost'});

figure
subplot(1,2,1);
boxplot(accuracyNAIVE1);
xlabel('Naive');
ylabel('Prediction Accuracy');
subplot(1,2,2);
boxplot(accuracyADA);
ylabel('Prediction Accuracy');
xlabel('AdaBoost');
set(gca,'xtick');
set(gcf,'NextPlot','add');
axes;
h = title('Comparison of Prediction Accuracies');
set(gca,'Visible','off');
set(h,'Visible','on');

figure
normplot(accuracyADA);
title('Normal Probability Plot: AdaBoost');
grid;

figure
normplot(accuracyNAIVE1);
title('Normal Probability Plot: Naive');
grid;

[H,pValue] = ttest2(accuracyADA, accuracyNAIVE1, 'tail', 'right');

%% extra