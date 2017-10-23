imagedata = loadMNISTImages('train-images.idx3-ubyte');
labeldata = loadMNISTLabels('train-labels.idx1-ubyte');

images = imagedata(:, 1:20000);
images = [images; ones(1, 20000)];
labels = labeldata(1:20000,:);
labels1hot = zeros(20000, 10);
for i=1:10
    labels1hot(labels == i - 1, i) = 1;
end


testimagedata = loadMNISTImages('t10k-images.idx3-ubyte');
testlabeldata = loadMNISTLabels('t10k-labels.idx1-ubyte');

testimages = testimagedata(:, 1:2000);
testimages = [testimages; ones(1, 2000)];
testlabels = testlabeldata(1:2000, :);
testlabels1hot = zeros(2000, 10);
for i=1:10
    testlabels1hot(testlabels == i - 1, i) = 1;
end

hidden_layer_units = 128;

w_hidden = rand(785, hidden_layer_units);
w_output = rand(hidden_layer_units, 10);

image = images(:, 1);

a_hidden = w_hidden' * image;

g_hidden = 1 ./ (1 + exp(a_hidden * -1));

a_output = g_hidden' * w_output;

g_output = a_output;
g_output(g_output < 0) = 0;

[value, indices] = max(g_output');
predictions = indices;
y = zeros(10, 1);
for i=1:10
    y(i, predictions == i) = 1;
end

y = y';
label = labels1hot(1, :);
delta_output = label - y;


syms a;
activ_func = 1 / (1 + exp(-a));
gradient_func = diff(activ_func);

delta_hidden = zeros(128, 1);
for i=1:128  
    a = a_hidden(i);
    delta_hidden(i) = sum(w_output(i, :) .* delta_output) * eval(gradient_func);
end

alpha = .1;
for i=1:128
    for j=1:10
        w_output(i, j) = w_output(i, j) + alpha * delta_output(j) * g_hidden(i);
    end
end

for i = 1:758
    for j = 1: 128
        w_hidden(i, j) = w_hidden(i, j) + alpha * delta_hidden(j) * image(i);
    end
end




