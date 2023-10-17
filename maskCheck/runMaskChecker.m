% Instantiate the MaskChecker class
checker = MaskChecker('maskCheckData');

% Load the data
checker = checker.loadData();

% Prepare the network (GoogleNet)
checker = checker.prepareNetwork();

% Display a random sample image from the training set (optional)
checker.displaySampleImage();

% Perform transfer learning
checker = checker.transferLearning();

% Validate the model and display accuracy
accuracy = checker.validate();
fprintf('Validation Accuracy: %f\n', accuracy);

% Display specific validation images (optional)
checker.displayValidationImages();
