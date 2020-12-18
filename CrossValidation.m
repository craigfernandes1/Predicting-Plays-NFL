%% Cross Validation Testing

T = readtable('FinalAllSeasonData.csv','Delimiter',',');
[n,~] = size(T);

tic
accuracyNAIVE1 = zeros(20,1);
accuracyNAIVE2 = zeros(20,1);
accuracyCART = zeros(20,1);
accuracyFOREST = zeros(20,1);
accuracyADA = zeros(20,1);
accuracyGRADIENT = zeros(20,1);
accuracyLOGISTIC = zeros(20,1);
accuracyKNN = zeros(20,1);

for ii = 1:2
    CVO = cvpartition(T.PlayType,'KFold',10);
    for jj = 1:CVO.NumTestSets      
        
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

%% box plot

figure
boxplot( [naive2Average,adaAverage, cartAverage, forestAverage,knnAverage, logAverage ],'color', 'rb','Labels',{'naive2','ada', 'cart', 'forest','knn', 'logistic'}, 'Widths', 0.5);
grid
title('Comparison of Prediction Accuracy'); 
ylabel('Prediction Accuracy');
xlabel('Model');
set(gca,'xtick');

figure
boxplot( [adaAverage, cartAverage, forestAverage,knnAverage, logAverage],'color', 'rb','Labels',{'AdaBoosting', 'CART', 'Random Forest','KNN', 'Logistic Regression'}, 'Widths', 0.5);
grid
title('Comparison of Prediction Accuracy'); 
ylabel('Prediction Accuracy');
xlabel('Model');
set(gca,'xtick');

figure
boxplot( [accuracies(1:30), cartAverage],'color', 'rb','Labels',{'Neural Net','CART'}, 'Widths', 1);
grid
title('Comparison of Prediction Accuracy'); 
ylabel('Prediction Accuracy');
xlabel('Model');
set(gca,'xtick');


figure
subplot(1,2,1);
boxplot(accuracies);
xlabel('Neural Net');
ylabel('Prediction Accuracy');
subplot(1,2,2);
boxplot( [cartAverage],'color', 'rb','Labels',{'CART'}, 'Widths', 0.5);
ylabel('Prediction Accuracy');
xlabel('AdaBoost');
set(gca,'xtick');
set(gcf,'NextPlot','add');
axes;
h = title('Comparison of Prediction Accuracies');
set(gca,'Visible','off');
set(h,'Visible','on');

figure
subplot(1,2,1);
boxplot(naive2Average);
xlabel('Naive');
ylabel('Prediction Accuracy');
subplot(1,2,2);
boxplot( [cartAverage],'color', 'rb','Labels',{'cart'}, 'Widths', 0.5);ylabel('Prediction Accuracy');
xlabel('AdaBoost');
set(gca,'xtick');
set(gcf,'NextPlot','add');
axes;
h = title('Comparison of Prediction Accuracies');
set(gca,'Visible','off');
set(h,'Visible','on');

%minitab plots are better
figure
normplot(cartAverage);
title('Normal Probability Plot: CART');
grid;

figure
normplot(accuracies);
title('Normal Probability Plot: Neural Net');
grid;


figure
plot = plotconfusion(actual',pred_CART')
title('CART Confusion Matrix')
xlabel('Target Class')
set(gca,'xticklabel',{'Rush' 'Pass' ''})
ylabel('Output Class')
set(gca,'yticklabel',{'Rush' 'Pass' ''})

figure
plot = plotconfusion(pred(1,:),T_Response(1,indices))
title('Neural Network Confusion Matrix')
xlabel('Target Class')
set(gca,'xticklabel',{'Rush' 'Pass' ''})
ylabel('Output Class')
set(gca,'yticklabel',{'Rush' 'Pass' ''})





