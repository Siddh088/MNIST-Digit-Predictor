import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Model:
    def __init__(self):
        self.home()

    def dataPreprocess(self, file):
        try:
            df = pd.read_csv(file)
            # Splitting the dataset into Dependent and independent Variables
            X = df.drop(columns=['label'])
            y = df['label']

            # Normalize the X data between 0-1
            if X.values.max() == 1.0:
                pass
            else:
                X = X / 255

            # Apply one-hot-encoding on Y data, convert to categorical problem
            y = to_categorical(y, 10)

            # Reshape the input data to 4-D (sampleSize, height, width, colorChannel)
            X = X.values.reshape(-1, 28, 28, 1)

            return X, y

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def home(self):
        user_input = input("""
                        Welcome, Which action would you want to perform?
                         1) Enter 1 to Train the model.
                         2) Enter 2 to Test the Accuracy of Model.
                         3) Enter 0 to [Exit] from the Application.
                           """)
        user_input = int(user_input)  # Convert user input to an integer

        if user_input == 1:
            try:
                file = str(input("Enter The training file: "))
                X_train, y_train = self.dataPreprocess(file)

                # Apply model
                trained_model, X_train, y_train = self.buildModel(X_train, y_train)

                # Save model
                trained_model.save('mnist_model.h5')

            except Exception as e:
                print(f"An error occurred: {str(e)}")

        elif user_input == 2:
            try:
                file = str(input("Enter The Test file: "))
                X_test, y_test = self.dataPreprocess(file)

                trained_model = load_model('mnist_model.h5')

                self.testingModel(X_test, y_test, trained_model)

            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print("Please Enter Correct choice.")

    def buildModel(self, X, y):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(4, 4),activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPool2D())
        model.add(Flatten())  # Flatten the output for fully connected layers
        model.add(Dense(128, activation='relu'))
        # Output layer
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

        early_stop = EarlyStopping(patience=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train, epochs=50, validation_data=(
            X_test, y_test), callbacks=[early_stop])
        print(f"Model Evaluate score is {model.evaluate(X_test, y_test)}")

        return model, X_train, y_train

    def testingModel(self, X_test, y_test, trained_model):
        # Evaluate the model on the test data
        evaluation = trained_model.evaluate(X_test, y_test)

        # Display the evaluation results
        print("Test Loss:", evaluation[0])
        print("Test Accuracy:", evaluation[1])

        # Predict on the test data
        y_pred = trained_model.predict(X_test)
        # Convert predictions to labels
        y_pred_labels = [np.argmax(pred) for pred in y_pred]

        # Convert one-hot encoded labels to class labels
        y_test_labels = [np.argmax(label) for label in y_test]

        # Evaluate accuracy
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        # Generate a classification report
        class_report = classification_report(y_test_labels, y_pred_labels)
        print("Classification Report:\n", class_report)

        # Create a confusion matrix
        confusion_mat = confusion_matrix(y_test_labels, y_pred_labels)
        print("Confusion Matrix:\n", confusion_mat)

# Instantiate and run the model
if __name__ == "__main__":
    model = Model()
