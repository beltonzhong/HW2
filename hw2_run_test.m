imagedata = loadMNISTImages('train-images.idx3-ubyte');
labeldata = loadMNISTLabels('train-labels.idx1-ubyte');

images = imagedata(:, 1:60000);
images = [images; ones(1, 60000)];
labels = labeldata(1:60000,:);
labels1hot = zeros(60000, 10);
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

w_hidden = saved_w_hidden;
w_output = saved_w_output;

for index = 1:2000
    image = testimages(:, index);
    a_hidden = w_hidden' * image;
    g_hidden = tanh(a_hidden);
    a_output = g_hidden' * w_output;
    sumexp = sum(exp(a_output));
    g_output = exp(a_output) ./ sumexp;

    [value, indices] = max(g_output');
    label = testlabels1hot(index, :);
    
    cost = 0;
    for k=1:10
        cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
    end
    loss = loss - cost;
    
    prediction = indices - 1;
    if prediction == testlabels(index)
        correctCount = correctCount + 1;
    end
end
disp(correctCount)
