%% �����
nx=[-3,-2,-1,0,1,2,3];
x1=[3 1 7 0 -1 4 2];
x2 = x1;
k = length(x2);
e = randn(1, k);
ny = nx + 2;
y = x2 + e;
% ʹ��xcorr����
figure();
subplot(1, 2, 1);
r1 = xcorr(x1, y);
nx_len = length(nx);
n = [nx(1)+ny(1): nx(nx_len)+ny(nx_len)];
stem(n, r1);
xlabel('x');
title('xcorr�����')
% ʹ��conv�����
subplot(1, 2, 2);
x1 = fliplr(x1);
conv_x1y = conv(y, x1);
conv_x1y = fliplr(conv_x1y);
stem(n, conv_x1y);
xlabel('x');
title('conv�����')

%% ��Ƶ����Ƶͼ
clc
clear all
fs=1000;
b=[1 0 0 0 1];
a=[1 0 0 0 -.8145];
[h,f]=freqz(b,a,512,fs);
mag=abs(h);%����
ph=angle(h);%��λ
subplot(2,1,1);
ph=ph*180/pi;%�ɻ���ת��Ϊ�Ƕ�
plot(f,mag);
grid;
xlabel('Frequency/Hz');
ylabel('Magnitude');
title('��Ƶ��Ӧ');
subplot(2,1,2);
plot(f,ph);
grid;
xlabel('Frequency/Hz');
ylabel('Phase');
title('��Ƶ��Ӧ');

%% ��̬����
clc
clear all
N=200;
n=linspace(-100,100,N);
x=sin(pi*n/2)+5*cos(pi*n);
N_fft=2^nextpow2(2*N);
w=linspace(0,2*pi,N_fft);
h_fft=(1+exp(-1j*4*w))./(1-0.8145*exp(-1j*4*w));
x_fft=fft(x,N_fft);
y_fft=x_fft.*h_fft;
y_temp=fftshift(ifft(y_fft));
y=y_temp(N_fft/2:N_fft/2+N-1);
figure;
plot(w,abs(h_fft),'b','LineWidth',2);
hold on;
plot(w,angle(h_fft),'g','LineWidth',2);
legend('����','��λ')
figure;
plot(n,x,'b');
hold on;
plot(n,real(y),'g');
legend('x(n)','y(n)��̬����')

%% ��ͨ�˲���
clc
clear all
fs=2000; %����Ƶ��
fc1=300;  %���޽�ֹƵ��
fc2=600;%���޽�ֹƵ��
N=20;     % �˲����Ľ���  
wlp=fc1/(fs/2);
whp=fc2/(fs/2);
wn=[wlp,whp];;   %�˲�����һ����������޽�ֹƵ��
%�ò�ͬ�ķ������N�׵��˲���
[b1 a1] = butter(N,wn, 'bandpass'); %butterworth
% 3dB��ͨ���Ʋ���40dB�����˥��
[b2 a2] = ellip(N,3, 40, wn,'bandpass')%��Բ
w3=hamming(N);  %������
b3=fir1(N-1,wn,w3); 
%����˲�����Ƶ����Ӧ
[H1 f1]=freqz(b1,a1);
[H2 f2]=freqz(b2,a2);
[H3 f3]=freqz(b3,1,512,fs);
figure;
subplot(2,1,1); 
plot(f1,20*log10(abs(H1)));
xlabel('Ƶ��/Hz');
ylabel('���/dB');  
title('butterworth�ķ�Ƶ����');
grid on;
subplot(2,1,2); 
plot(f1,180/pi*unwrap(angle(H1)));
xlabel('Ƶ��/Hz');
ylabel('��λ');  
title('butterworth����Ƶ����');
grid on;
figure;
subplot(2,1,1);
plot(f2,20*log10(abs(H2)));
xlabel('Ƶ��/Hz');
ylabel('���/dB');  
title('��Բ�ķ�Ƶ����');
grid on;
subplot(2,1,2); 
plot(f2,180/pi*unwrap(angle(H2)));
xlabel('Ƶ��/Hz');
ylabel('��λ');  
title('��Բ����Ƶ����');
grid on;
figure;
subplot(2,1,1);
plot(f3,20*log10(abs(H3)));
xlabel('Ƶ��/Hz');
ylabel('���/dB');  
title('�������ķ�Ƶ����');
grid on;
subplot(2,1,2);
plot(f3,180/pi*unwrap(angle(H3)));
xlabel('Ƶ��/Hz');
ylabel('��λ');  
title('����������Ƶ����');
grid on;