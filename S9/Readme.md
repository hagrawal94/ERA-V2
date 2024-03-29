<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>
    <h2>Key Insights</h2>
    <h3>Model Architecture and Efficiency</h3>
    <ul>
        <li><strong>Depthwise Separable Convolutions</strong>: Learned the importance of using depthwise separable convolutions for reducing the model's parameter count without sacrificing the ability to extract meaningful features from images.</li>
        <li><strong>Channel and Layer Optimization</strong>: Gained insights into balancing the number of channels and layers to manage model complexity and ensure efficiency.</li>
        <li><strong>Adaptive Pooling</strong>: Understood the utility of adaptive pooling in ensuring consistent output sizes, which is crucial for aligning the model's output with classification layers.</li>
    </ul>
    <h3>Data Augmentation and Regularization</h3>
    <ul>
        <li><strong>Albumentations Library</strong>: Explored the Albumentations library for advanced image augmentation techniques, appreciating its flexibility and effectiveness in enhancing model generalization.</li>
        <li><strong>Dropout and Batch Normalization</strong>: Confirmed the effectiveness of dropout in preventing overfitting, and recognized the role of batch normalization in stabilizing training.</li>
    </ul>
    <h3>Training Strategies</h3>
    <ul>
        <li><strong>Learning Rate Optimization</strong>: Acknowledged the importance of learning rate scheduling and adjustments to improve model convergence and performance.</li>
        <li><strong>Impact of Augmentation</strong>: Investigated how different data augmentation strategies can significantly affect model accuracy and robustness.</li>
    </ul>
    <h2>Challenges Encountered</h2>
    <p>Balancing the model's depth and width within the parameter constraints posed a significant challenge, requiring careful architectural planning. Ensuring the receptive field exceeded 44 while maintaining or improving model accuracy involved strategic adjustments to convolution and pooling layers.</p>
</body>
</html>
