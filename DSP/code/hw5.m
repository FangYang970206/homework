function [nn_1, nn_2, nn_3, test_x] = hw5()
load mnist_uint8; %导入mnist数据集

train_x = double(train_x) / 255; %归一化
test_x  = double(test_x)  / 255; %归一化
train_y = double(train_y);
test_y  = double(test_y);

%% 深度置信网络(dbn) + 神经网络(nn)
rand('state',0)
%训练dpn
dbn.sizes = [100 100]; %784->100->100->784
opts.numepochs =   1; %dpn epochs
opts.batchsize = 100;  % dpn batchsize
opts.momentum  =   0; % sgd no momentum
opts.alpha     =   1;  %dpn参数
dbn = dbnsetup(dbn, train_x, opts); % 初始化dpn
dbn = dbntrain(dbn, train_x, opts); %训练dpn

%将深度置信网络参数导入神经网络
nn_1 = dbnunfoldtonn(dbn, 10);
nn_1.learningRate = 1;   %nn 学习率
nn_1.activation_function = 'sigm'; %nn激活函数
nn_1.output  = 'softmax'; %输出softmax
nn_1.weightPenaltyL2 = 1e-4; %权重二范数惩罚
opts.numepochs =  3; %nn epochs
opts.batchsize = 100; %nn batchsize
% opts.plot = 1;  %使能画图
%训练nn
nn_1 = nntrain(nn_1, train_x, train_y, opts);
%测试nn
[er1, bad1] = nntest(nn_1, test_x, test_y);%测试
fprintf('dpn+nn test error: %f\n', er1);
%% 神经网络(nn)
rand('state',0)
nn_2 = nnsetup([784 100 10]); %784->100->10
nn_2.learningRate = 1;   %nn 学习率
nn_2.activation_function = 'sigm';%nn激活函数
nn_2.output  = 'softmax'; %输出softmax
nn_2.weightPenaltyL2 = 1e-4; %权重二范数惩罚
opts.numepochs =  3; %nn epochs
opts.batchsize = 100; %nn batchsize
% opts.plot = 1; %使能画图
%训练nn
[nn_2, L] = nntrain(nn_2, train_x, train_y, opts);
%测试nn
[er2, bad2] = nntest(nn_2, test_x, test_y);
fprintf('nn2 test error: %f\n', er2);

%% 堆栈式自编码器(sae)+神经网络(nn)
rand('state',0)
sae = saesetup([784 100]); %784->100->100->784
sae.ae{1}.activation_function  = 'sigm'; %激活函数
sae.ae{1}.learningRate  = 1;  %学习率
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   3;%epochs
opts.batchsize = 100;%batchsize
%训练sae
sae = saetrain(sae, train_x, opts);
%设置nn结构
nn_3 = nnsetup([784 100 10]); %784->100->10
nn_3.activation_function    = 'sigm';%激活函数
nn_3.learningRate  = 1;%学习率
nn_3.output  = 'softmax'; %输出softmax
nn_3.weightPenaltyL2 = 1e-4;%权重二范数惩罚
nn_3.W{1} = sae.ae{1}.W{1};%初始化nn权重
opts.numepochs =   3;%nn epochs
opts.batchsize = 100;%nn batchsize
% opts.plot = 1; %使能画图
%训练nn
nn_3 = nntrain(nn_3, train_x, train_y, opts);
%测试nn
[er3, bad3] = nntest(nn_3, test_x, test_y);
fprintf('sae+nn test error: %f\n', er3);
end