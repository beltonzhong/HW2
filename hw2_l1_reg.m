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

w_hidden = rand(785, hidden_layer_units) - .5;
w_output = rand(hidden_layer_units, 10) - .5;


numTrainingIterations = 0;
iterationCount = 0;
correctCount = 0;
numCorrect = zeros(1, 1);
loss = 0;
lossValues = zeros(1, 1);

vCorrectCount = 0;
vNumCorrect = zeros(1, 1);
vLoss = 0;
vLossValues = zeros(1, 1);

lambda = .001;
while 1
    index = int32(mod(numTrainingIterations, 50000));
    image = images(:, index + 1);
    a_hidden = w_hidden' * image;
    g_hidden = tanh(a_hidden);
    a_output = g_hidden' * w_output;
    sumexp = sum(exp(a_output));
    g_output = exp(a_output) ./ sumexp;

    [value, indices] = max(g_output');
    label = labels1hot(index + 1, :);
    delta_output = label - g_output;
    
    cost = 0;
    for k=1:10
        cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
    end
    loss = loss - cost;
    
    prediction = indices - 1;
    if prediction == labels(index + 1)
        correctCount = correctCount + 1;
    end

    delta_hidden = zeros(1, hidden_layer_units);
    for i=1:hidden_layer_units
        a = a_hidden(i);
        grad = 1 - (tanh(a) ^ 2);
        delta_hidden(i) = sum(w_output(i, :) .* delta_output) * grad;
    end

    lambda = .0001;
    alpha = .01;  
    w_output = w_output + alpha * (g_hidden * delta_output + lambda * sign(w_output));    
    w_hidden = w_hidden + alpha * (image * delta_hidden + lambda * sign(w_hidden));
    
    iterationCount = iterationCount + 1;
    if iterationCount == 1000
        numCorrect(int32(floor(numTrainingIterations / 1000)) + 1) = correctCount;
        lossValues(int32(floor(numTrainingIterations / 1000)) + 1) = loss / 1000;
        correctCount = 0;
        loss = 0;
        iterationCount = 0;
        
        for vItCount = 1:10000
            vImage = images(:, 50000 + vItCount);
            a_hidden = w_hidden' * vImage;
            g_hidden = tanh(a_hidden);
            a_output = g_hidden' * w_output;
            sumexp = sum(exp(a_output));
            g_output = exp(a_output) ./ sumexp;

            [value, indices] = max(g_output');
            
            label = labels1hot(50000 + vItCount, :);
            cost = 0;
            for k=1:10
                cost = cost + (label(k) * log(g_output(k)) + (1 - label(k)) * log(1 - g_output(k)));
            end
            loss = loss - cost;

            prediction = indices - 1;
            if prediction == labels(vItCount + 50000)
                correctCount = correctCount + 1;
            end
        end
        vNumCorrect(int32(floor(numTrainingIterations / 1000)) + 1) = correctCount;
        vLossValues(int32(floor(numTrainingIterations / 1000)) + 1) = loss / 1000;
        correctCount = 0;
        loss = 0;
        if (size(vNumCorrect, 2) > 2) && (abs(vNumCorrect(end) - vNumCorrect(end - 1)) < 1)
            break;
        end
    end
    numTrainingIterations = numTrainingIterations + 1;
end