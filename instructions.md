How to Use the test_api.py Script
Open the Script: Open the test_api.py file in your code editor (like VS Code).

Modify the Input Data: Inside the script, you will find a Python dictionary called user_data. This dictionary contains all the health parameters your model needs to make a prediction.

Edit the Values: Change the values in the user_data dictionary to represent the person whose life expectancy you want to predict. You can test any scenario by changing values like Age, Smoking status, Blood Pressure, Daily Activity, etc.

Save the File: After making your changes, save the test_api.py file.

Run the Script:

First, make sure your Flask server (app.py) is still running in your first terminal.

In your second terminal (the one where you activated the venv-webapp environment and installed requests), run the following command:

python test_api.py

View the Result: The script will print the prediction it receives back from your server directly in the terminal. You can repeat this process as many times as you like with different data to see how the predictions change.