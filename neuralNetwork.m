%Read in data

T = readtable('FinalAllSeasonData.csv','Delimiter',',');

[n,~] = size(T);

parameters = 1;
data = zeros(21,3);
index = 1;
        
            T_Response = table2array(T(:,8));

            temp = find(T_Response == 0);
            T_Response(temp,2) = 1;
            temp = find(T_Response(:,1) == 1);
            T_Response(temp,2) = 0;

            inputs = [T.Quarter'; T.Minute'; 
                      T.Second'; T.Down'; 
                      T.ToGo'; T.YardLine';
                      T.PointDiff';];

            T_Response = T_Response';
            
            
            for jj = 1:2
                k = 1;
                for ii = 10:2:26
                    net = patternnet([ii]);
                    [net, modelData] = train(net,inputs,T_Response);
                    indices = modelData.testInd;

                    pred = net(inputs(:,indices));
                    pred = round(pred);

                    accuracy = 1-sum(abs(pred(1,:)-T_Response(1,indices)))/(n*0.15);
                    accuracies2(jj,k) = accuracy;
                    k = k+1
                end
            end