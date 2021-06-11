clc
clear

%row = type of beamformer, collumn = SNR (-20, -10, -5, 0, 5)
ESTOI = [0.044 0.248 0.390 0.532 0.659 0.760;...
    0.024 0.108 0.155 0.196 0.221 0.228;...
    0.026 0.058  0.118 0.190 0.249 0.285;...
    0.058 0.235 0.324 0.398 0.451 0.483]; 

SNR = [-0.094 9.374 14.837 21.685 27.603 33.096;...
    -0.697 9.564 17.114 23.932 29.928 35.146;...
    -0.366 9.107 14.804 21.388 27.516 33.124;...
    -1.288 8.411 13.667 21.646 28.107 33.774];

figuresFolder = dir('SNR/*.fig');

for i = 1:length(figuresFolder)
    figureName = figuresFolder(i).name;
    fig = openfig(figureName);
    figures{i} = fig;
    data(i,:) = get(get(gca, 'Children'),'YData');
    close(fig);
end

figure;
t = tiledlayout(3,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile;
plot (data{2,4});
hold on;
plot (data{2,3});
hold on;
plot (data{2,2});
hold on;
plot (data{2,1});
legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
title ('Segmental SNR for input SNR = -20 dB');

nexttile;
plot (data{1,4});
hold on;
plot (data{1,3});
hold on;
plot (data{1,2});
hold on;
plot (data{1,1});
legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
title ('Segmental SNR for input SNR = -10 dB');

nexttile;
plot (data{3,4});
hold on;
plot (data{3,3});
hold on;
plot (data{3,2});
hold on;
plot (data{3,1});
legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
title ('Segmental SNR for input SNR = -5 dB');

nexttile;
plot (data{4,4});
hold on;
plot (data{4,3});
hold on;
plot (data{4,2});
hold on;
plot (data{4,1});
legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
title ('Segmental SNR for input SNR = 0 dB');

nexttile;
plot (data{6,4});
hold on;
plot (data{6,3});
hold on;
plot (data{6,2});
hold on;
plot (data{6,1});
legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
title ('Segmental SNR for input SNR = 5 dB');

nexttile;
plot (data{5,4});
hold on;
plot (data{5,3});
hold on;
plot (data{5,2});
hold on;
plot (data{5,1});
legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
title ('Segmental SNR for input SNR = 10 dB');


figure;
plot(ESTOI');
legend({'DSB','MVDR', 'MVDR ml_d', 'MWF'},'Location','northwest');
xticklabels({'-20dB','-10dB','-5dB','0dB','+5dB', '+10dB'});
xticks([1 2 3 4 5 6]);
title('ESTOI for different beamformers');
