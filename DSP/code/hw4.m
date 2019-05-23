%% ������
clc;
close all;
clear all;
% ÿ���������Ϊ10
N = 10;
%�����ʼ�������������
data1 = rand(2, 10);
data2 = rand(2, 10);
%�����������label
data1_label = zeros(1,10);
data2_label = ones(1,10);
%������������ݷֿ�
data1(1,:) = data1(1,:) - 0.5;
data1(2,:) = data1(2,:) + 0.5;
data2(1,:) = data2(1,:) + 0.5;
data2(2,:) = data2(2,:) - 0.5;
%���data��label
data = [data1, data2];
label = [data1_label, data2_label];
% �����֪����Ԫ�������ʼ�� 
net=newp([0 1;0 1],1);
net.initFcn='initlay';
net.layers{1}.initFcn='initwb';
net.inputWeights{1,1}.initFcn='rands';
net.layerWeights{1,1}.initFcn='rands';
net.biases{1}.initFcn='rands';
net=init(net);
% ѵ����֪����Ԫ
net=train(net,data,label);
cell2mat(net.iw);
cell2mat(net.b);
%���ƽ��
figure();
scatter(data1(1,:), data1(2,:));
hold on;
scatter(data2(1,:), data2(2,:));
plotpc(net.iw{1,1},net.b{1});

%% �ع�
clc;
clear all;
% ����ѵ���������������
x1= 0:0.5:4*pi;
x2= 0:0.12:4*pi;
% P1 = (x1.^2-2*x1).*exp(-x1.^2-x2.^2-x1.*x2); 
P1 = 0.12*exp(-0.23*x1) + 0.54*exp(-0.17*x1).*sin(1.23*x1);% ѵ������
T1 = P1; % ѵ��Ŀ��
P2 = 0.12*exp(-0.23*x2) + 0.54*exp(-0.17*x2).*sin(1.23*x2);% ��������
T2 = P2; % ����Ŀ��
% ��һ��
[PN1,minp,maxp,TN1,mint,maxt] = premnmx(P1,T1);
PN2 = tramnmx(P2,minp,maxp);
TN2 = tramnmx(T2,mint,maxt);
% �����������
HideNum=1;   % �������
NodeNum = 5; % ����ڵ��� 
TypeNum = 1; % ���ά��
TF1 = 'tansig';TF2 = 'purelin'; % �б���(ȱʡֵ)
net = newff(minmax(PN1),[NodeNum TypeNum],{TF1 TF2});
net.trainFcn = 'trainlm'; 
net.trainParam.show = 20; % ѵ����ʾ���
net.trainParam.lr = 0.3; % ѧϰ���� - traingd,traingdm
net.trainParam.mc = 0.95; % ������ϵ�� - traingdm,traingdx
net.trainParam.mem_reduc = 1; % �ֿ����Hessian����net.trainParam.epochs = 1000; % ���ѵ������
net.trainParam.goal = 1e-4; % ��С�������
net.trainParam.min_grad = 1e-20; % ��С�ݶ�
net.trainParam.time = inf; % ���ѵ��ʱ��
net = train(net,PN1,TN1); % ѵ��
YN1 = sim(net,PN1); % ѵ������ʵ�����
YN2 = sim(net,PN2); % ��������ʵ�����
MSE1 = mean((TN1-YN1).^2) % ѵ���������
MSE2 = mean((TN2-YN2).^2) % ���Ծ������
Y2 = postmnmx(YN2,mint,maxt); % ����һ��
% �����ͼ
plot(1:length(T2),T2,'r',1:length(Y2),Y2,'b');
fprintf('���Ծ������Ϊ��%f\n', MSE2);
legend('���Լ�','ѵ����'); 
title('���������ʵ������Ա�');

