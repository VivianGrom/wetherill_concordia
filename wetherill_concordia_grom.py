#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:10:17 2023

@author: viviangrom
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tkinter import Tk, filedialog

# Define decay constants for U-235 and U-238
lambda_235U = 9.8485e-10  # Decay constant for U-235 (per year)
lambda_238U = 1.55125e-10  # Decay constant for U-238 (per year)

# Define a function to calculate ages based on 207Pb/235U and 206Pb/238U ratios
def calculate_age(r207Pb235U, r206Pb238U):
    age = (1 / lambda_235U) * np.log(1 + (r207Pb235U / r206Pb238U))
    return age

# Create a Tkinter root window
root = Tk()
root.withdraw()  # Hide the root window

# Open a file explorer dialog to select the Excel file
file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])

# Check if a file is selected
if file_path:
    # Use pandas to read the Excel file into a DataFrame
    df = pd.read_excel(file_path)

pb207_u235 = df['207Pb/235U']
pb206_u238 = df['206Pb/238U']
index = df['index']

# Calculate ages for the data points
ages = []
for i in range(len(df)):
    age = calculate_age(df['207Pb/235U'].iloc[i], df['206Pb/238U'].iloc[i])
    ages.append(age)

# Define a function to calculate the Concordia curve
def concordia_curve(t):
    X = (np.exp(lambda_235U * t) - 1)
    Y = (np.exp(lambda_238U * t) - 1) 
    return X, Y

# Create a range of time values (in Ma)
t_values = np.linspace(0, 3600000000, 100)  # Adjust the time range as needed

# Calculate the corresponding X and Y values for the Concordia curve
X_concordia, Y_concordia = concordia_curve(t_values)

X_plot = []
Y_plot = []
a_values = []
a = 0

for i in range(8):
   X, Y = concordia_curve(a)
   X_plot.append(X)
   Y_plot.append(Y)
   a_values.append(a/1000000000)
   a = a + 500000000

# Group the data by the color values (third column)
grouped_data = df.groupby('index')

# Create a dictionary to store regression results for each group
regressions = {}
slope_tot = []
intercept_tot = []

# Perform linear regression for each group
for color, group in grouped_data:
    x_group = group['207Pb/235U']
    y_group = group['206Pb/238U']
    
    # Create and fit a linear regression model
    model = LinearRegression()
    model.fit(x_group.values.reshape(-1, 1), y_group)
    
    # Get the slope (coefficient) and intercept
    slope = model.coef_[0]
    slope_tot.append(slope)
    intercept = model.intercept_
    intercept_tot.append(intercept)
    
    # Store the regression model in the dictionary
    regressions[color] = model

# Plot the Wetherill Concordia diagram with data points and Concordia curve
plt.figure(figsize=(12, 8))

for i, (x, y, a) in enumerate(zip(X_plot, Y_plot, a_values)):
    plt.text(x, y, f'{a} Ga', fontsize=10, ha='right', va='bottom')

# Customize labels, axis limits, and legend as needed
plt.xlabel('207Pb/235U')
plt.ylabel('206Pb/238U')

# Plot the Concordia curve
plt.plot(X_concordia, Y_concordia, color='black', label='Concordia Curve')
plt.plot(X_plot, Y_plot, 'o', color='black')

# Plot data points
#plt.scatter(df['207Pb/235U'], df['206Pb/238U'], c=index, cmap='winter', label='Data Points', s=50)

for color, group in grouped_data:
    x_group = group['207Pb/235U']
    y_group = group['206Pb/238U']
    model = regressions[color]
    
    # Plot data points for the group
    plt.scatter(x_group, y_group, label=f'Group: {color}')
    
    # Plot the regression line
    y_pred = model.predict(x_group.values.reshape(-1, 1))
    plt.plot(x_group, y_pred, linewidth=2)

# Show the plot
plt.grid(True)
plt.legend()
plt.show()

# Form the equations
equation1 = f'y = {slope_tot[0]:.4f}x + {intercept_tot[0]:.4f}'
equation2 = f'y = {slope_tot[1]:.4f}x + {intercept_tot[1]:.4f}'
equation3 = f'y = {slope_tot[2]:.4f}x + {intercept_tot[2]:.4f}'
equation4 = f'y = {slope_tot[3]:.4f}x + {intercept_tot[3]:.4f}'
print('Linear Regression Event 1:', equation1)
print('Linear Regression Event 2:', equation2)
print('Linear Regression Event 3:', equation3)
print('Linear Regression Event 4:', equation4)







