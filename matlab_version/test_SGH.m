function test_SGH()
addpath('utils');
[trainL, trainX, testL, testX] = load_data();

train_norm = sqrt(sum(trainX.*trainX, 2));
trainX = bsxfun(@rdivide, trainX, train_norm);

test_norm = sqrt(sum(testX.*testX, 2));
testX = bsxfun(@rdivide, testX, test_norm);

num_training = size(trainX,1);
num_testing = size(testX,1);
bit = 64;
% Kernel parameter
m = 300;
sample = randperm(num_training);
bases = trainX(sample(1:m),:);

%% Training procedure
disp('start training');
rho = 2;
[Wx,KXTrain,para] = trainSGH(trainX,bases,bit, rho);
disp('end training');

trainB = compactbit(KXTrain*Wx > 0);

%% Testing procedure
% construct KXTest
KTest = distMat(testX,bases);
KTest = KTest.*KTest;
KTest = exp(-KTest/(2*para.delta));
KXTest = KTest-repmat(para.bias,num_testing,1);

testB = compactbit(KXTest*Wx > 0);

Dhamm = hammingDist(testB, trainB);

Wtrue = testL * trainL';
disp('start evaluating');

map = callMap(Wtrue,Dhamm);
disp(['map: ' num2str(map, '%4f')]);
end

function [trainL, trainX, testL, testX] = load_data()
filename = 'MNIST.h5';
trainX = double(h5read(filename, '/XDatabase'))';
trainL = double(h5read(filename, '/databaseL'))';
testX = double(h5read(filename, '/XTest'))';
testL = double(h5read(filename, '/testL'))';

mean_ = mean(trainX);
trainX = bsxfun(@minus, trainX, mean_);
testX = bsxfun(@minus, testX, mean_);

end