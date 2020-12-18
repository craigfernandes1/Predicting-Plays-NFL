%load the play-by-play for SuperBowl 
T = readtable('Game2012Test4.csv','Delimiter',',');
[n,~] = size(T)

inputs = [T.Quarter'; T.Minute'; 
                      T.Second'; T.Down'; 
                      T.ToGo'; T.YardLine';
                      T.PointDiff';];

T_Response = table2array(T(:,8));

temp = find(T_Response == 0);
T_Response(temp,2) = 1;
temp = find(T_Response(:,1) == 1);
T_Response(temp,2) = 0;

T_Response = T_Response';


%load the model
% net = load('NETmodel.mat');

pred_NET = net(inputs);
pred_NET = round(pred_NET);
NETaccuracy = 1-sum(abs(pred_NET(1,:)-T_Response(1,:)))/n


%predict and compute the accuracy
CART = load('CARTmodel4.mat');
pred_CART = predict(CART.CART, T);                                            
Cartaccuracy = 1-sum(abs(pred_CART-T.PlayType))/n

pred_naive = ones(n,1);
NAIVEaccuracySB = 1-sum(abs(pred_naive-T.PlayType))/n