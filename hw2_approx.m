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

for counter = 1:20000
    image = images(:, counter);

    a_hidden = w_hidden' * image;

    g_hidden = 1 ./ (1 + exp(a_hidden * -1));

    a_output = g_hidden' * w_output;
    
    sumexp = sum(exp(a_output));
    g_output = exp(a_output) ./ sumexp;

    [value, indices] = max(g_output');
    predictions = indices;
    y = zeros(10, 1);
    for i=1:10
        y(i, predictions == i) = 1;
    end

    y = y';
    label = labels1hot(counter, :);
    delta_output = label - y;

    delta_hidden = zeros(hidden_layer_units, 1);
    for i=1:hidden_layer_units
        a = a_hidden(i);
        grad = exp(-a) / ((1 + exp(-a)) ^ 2);
        delta_hidden(i) = sum(w_output(i, :) .* delta_output) * grad;
    end

    alpha = .1;
    for i=1:hidden_layer_units
        for j=1:10
            w_output(i, j) = w_output(i, j) + alpha * delta_output(j) * g_hidden(i);
        end
    end

    for i = 1:758
        for j = 1: 128
            w_hidden(i, j) = w_hidden(i, j) + alpha * delta_hidden(j) * image(i);
        end
    end
end

epsilon = .01;
image = images(:, 1);

grad_approx_o = zeros(hidden_layer_units, 10);
grad_approx_o = grad_approx_o - 1;
for i=1:hidden_layer_units
    for j=1:10
        w = w_output(i, j);
        bpg = g_hidden(i) * delta_output(j);
        
        wh = w + epsilon;
        wl = w - epsilon;
        
        w_output(i, j) = wh;
        %foward prop on wh
        a_hidden = w_hidden' * image;

        g_hidden = 1 ./ (1 + exp(a_hidden * -1));

        a_output = g_hidden' * w_output;

        sumexp = sum(exp(a_output));
        g_output = exp(a_output) ./ sumexp;

        label = labels1hot(counter, :);
        
        cost = 0;
        for k=1:10
            cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
        end
        Hcost = -1 * cost;
        
        w_output(i, j) = wl;
        %forward prop on wl
         a_hidden = w_hidden' * image;

        g_hidden = 1 ./ (1 + exp(a_hidden * -1));

        a_output = g_hidden' * w_output;

        sumexp = sum(exp(a_output));
        g_output = exp(a_output) ./ sumexp;

        label = labels1hot(counter, :);
        
        cost = 0;
        for k=1:10
            cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
        end
        Lcost = -1 * cost;
        
        grad_approx_o (i, j) = (Hcost - Lcost) / 2 * epsilon;
    end
end

grad_approx_h = zeros(785, hidden_layer_units);
grad_approx_h = grad_approx_h - 1;
for i = 1:785
    for j = 1:hidden_layer_units
        w = w_hidden(i, j);
        bpg = image(i) * delta_hidden(j);
        wh = w + epsilon;
        wl = w - epsilon;
        
        w_hidden(i, j) = wh;
        %foward prop on wh
        a_hidden = w_hidden' * image;

        g_hidden = 1 ./ (1 + exp(a_hidden * -1));

        a_output = g_hidden' * w_output;

        sumexp = sum(exp(a_output));
        g_output = exp(a_output) ./ sumexp;
        
        label = labels1hot(counter, :);
        
        cost = 0;
        for k=1:10
            cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
        end
        Hcost = -1 * cost;
        
        w_hidden(i, j) = wl;
        %forward prop on wl
         a_hidden = w_hidden' * image;

        g_hidden = 1 ./ (1 + exp(a_hidden * -1));

        a_output = g_hidden' * w_output;

        sumexp = sum(exp(a_output));
        g_output = exp(a_output) ./ sumexp;

        label = labels1hot(counter, :);
        
        cost = 0;
        for k=1:10
            cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
        end
        Lcost = -1 * cost;
        
        grad_approx_h(i, j) = (Hcost - Lcost) / 2 * epsilon;   
    end
end