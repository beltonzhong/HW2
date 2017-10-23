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

w_hidden = rand(785, hidden_layer_units);
w_output = rand(hidden_layer_units, 10);

loss = zeros(60000, 1);
correct = 0;
rates = zeros(10, 1);
for epochCount=1:20
    for counter = 1:60000
        image = images(:, counter);

        a_hidden = w_hidden' * image;

        g_hidden = 1 ./ (1 + exp(a_hidden * -1));

        a_output = g_hidden' * w_output;

        sumexp = sum(exp(a_output));
        g_output = exp(a_output) ./ sumexp;

        [value, indices] = max(g_output');
        prediction = indices - 1;
        if prediction == labels(counter)
            correct = correct + 1;
        end

        label = labels1hot(counter, :);

        cost = 0;
        for k=1:10
            cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
        end
        loss(counter) = -cost;

        delta_output = label - g_output;

        delta_hidden = zeros(1, hidden_layer_units);
        for i=1:hidden_layer_units
            a = a_hidden(i);
            grad = exp(-a) / ((1 + exp(-a)) ^ 2);
            delta_hidden(i) = sum(w_output(i, :) .* delta_output) * grad;
        end

        lambda = .0001;
        alpha = .01;  
        w_output = w_output + alpha * (g_hidden * delta_output + lambda * w_output);    
        w_hidden = w_hidden + alpha * (image * delta_hidden + lambda * w_hidden);
    end
    rates(epochCount) = correct / 600;

end