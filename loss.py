import re

# Regex pattern to find "Training loss" and "Validation Loss" and their values
pattern = re.compile(r'^Training loss: ([\d.e-]+), Validation Loss: ([\d.e-]+)')

# Lists to hold the training and validation loss values
training_losses = []
validation_losses = []

# Open and read the log file
with open("output.log", 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            # Extract training and validation loss values
            training_loss, validation_loss = match.groups()
            training_losses.append(float(training_loss))
            validation_losses.append(float(validation_loss))

# Optionally, print the lists
print("Training Losses:", training_losses)
print("Validation Losses:", validation_losses)

print(len(training_losses))