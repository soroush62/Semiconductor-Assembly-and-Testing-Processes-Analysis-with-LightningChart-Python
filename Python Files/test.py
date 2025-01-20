# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# import lightningchart as lc
# import time

# # Load license key
# with open(
#     "D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt", "r"
# ) as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load the dataset
# data = pd.read_csv("Dataset/mixed_categorical_numerical_data.csv")

# # Separate the target variable (Y) and features (X)
# target = data["Y"]
# features = data.drop(columns=["Y"])

# # Identify categorical and numerical columns
# categorical_columns = ["X1", "X2", "X3", "X4", "X5"]
# numerical_columns = [col for col in features.columns if col not in categorical_columns]

# # One-hot encode the categorical variables
# encoder = OneHotEncoder(sparse_output=False, drop="first")
# encoded_categorical = encoder.fit_transform(features[categorical_columns])

# # Combine encoded categorical variables with numerical ones
# encoded_features = np.hstack((encoded_categorical, features[numerical_columns].values))

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     encoded_features, target, test_size=0.2, random_state=42
# )

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict and calculate residuals on the test set
# y_pred = model.predict(X_test)
# residuals = np.abs(y_test - y_pred)

# # Shift y-values to make all values positive
# shift_value = abs(min(y_test.min(), y_pred.min(), residuals.min()))
# y_test_shifted = y_test + shift_value
# y_pred_shifted = y_pred + shift_value
# residuals_shifted = residuals + shift_value

# # Normalize residuals to derive certainty values (1 - normalized residuals)
# certainty = 1 - (residuals_shifted / residuals_shifted.max())
# certainty_values = certainty.values  # Ensure we have values for visualization

# # Parameters for heatmap visualization
# num_rows = 100
# time_delay = 0.2
# batch_size = 10  # Steps to process per batch
# line_chart_delay = 30  # 2-second delay for line chart

# # Determine the full range of y-axis values (spanning actual and predicted)
# y_min = min(y_test_shifted.min(), y_pred_shifted.min())
# y_max = max(y_test_shifted.max(), y_pred_shifted.max())
# y_range = np.linspace(y_min, y_max, num_rows)


# # Generate heatmap matrix based on prediction range
# def generate_certainty_heatmap(y_pred_value, certainty_value):
#     """
#     Generate a single column of certainty heatmap based on the prediction range and certainty value.
#     """
#     column = 1 - np.abs(y_range - y_pred_value) / (
#         y_max - y_min
#     )  # Intensity based on proximity
#     column = column * certainty_value  # Scale by certainty
#     column = np.clip(column, 0, 1)  # Normalize to [0, 1]
#     return column


# # Initialize heatmap matrix
# columns = len(y_pred_shifted)
# mat = np.zeros((num_rows, columns))

# # Create a LightningChart instance
# chart = lc.ChartXY(theme=lc.Themes.Dark)

# # Add a heatmap grid series
# heatmap_series = chart.add_heatmap_grid_series(
#     columns=columns, rows=num_rows, data_order="rows"
# ).set_name("Real-Time Certainty Heatmap")
# heatmap_series.set_step(
#     x=(len(y_pred_shifted) - 1) / columns,  # Match the time step
#     y=(y_max - y_min) / num_rows,  # Scale y step to span the prediction range
# )

# # Set the color palette for the heatmap
# heatmap_series.set_palette_coloring(
#     steps=[
#         {
#             "value": 0.0,
#             "color": lc.Color(0, 0, 0, 128),
#         },  # Transparent for low certainty
#         {
#             "value": 0.25,
#             "color": lc.Color(255, 255, 0, 128),
#         },  # Yellow for medium certainty
#         {
#             "value": 0.5,
#             "color": lc.Color(255, 192, 0, 255),
#         },  # Orange for higher certainty
#         {"value": 1.0, "color": lc.Color(255, 0, 0, 255)},  # Red for maximum certainty
#     ],
#     look_up_property="value",
#     percentage_values=True,
#     interpolate=True,
# )

# # Customize heatmap appearance
# heatmap_series.hide_wireframe()
# heatmap_series.set_intensity_interpolation(True)

# # Add actual predictions as a line series
# prediction_series = chart.add_line_series()
# prediction_series.set_name("Predictions").set_line_color(
#     lc.Color(0, 128, 255)
# ).set_line_thickness(2)

# # Open the chart in live mode
# chart.open(live=True, method="browser")

# # Stream certainty values and predictions in batches
# for t in range(0, columns, batch_size):
#     # Update heatmap
#     for i in range(batch_size):
#         if t + i < columns:
#             mat[:, t + i] = generate_certainty_heatmap(
#                 y_pred_shifted[t + i], certainty_values[t + i]
#             )

#     # Invalidate the heatmap for the updated batch
#     heatmap_series.invalidate_intensity_values(mat)

#     # Delay the line chart update by 2 seconds (converts delay to steps)
#     time.sleep(time_delay)
#     if t >= int(line_chart_delay / time_delay):
#         for i in range(batch_size):
#             if t + i - int(line_chart_delay / time_delay) < columns:
#                 prediction_series.add(
#                     x=t + i - int(line_chart_delay / time_delay),
#                     y=y_pred_shifted[t + i - int(line_chart_delay / time_delay)],
#                 )

# chart.close()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# import lightningchart as lc
# import time

# # Load license key
# with open(
#     "D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt", "r"
# ) as f:
#     mylicensekey = f.read().strip()
# lc.set_license(mylicensekey)

# # Load the dataset
# data = pd.read_csv("Dataset/mixed_categorical_numerical_data.csv")

# # Separate the target variable (Y) and features (X)
# target = data["Y"]
# features = data.drop(columns=["Y"])

# # Identify categorical and numerical columns
# categorical_columns = ["X1", "X2", "X3", "X4", "X5"]
# numerical_columns = [col for col in features.columns if col not in categorical_columns]

# # One-hot encode the categorical variables
# encoder = OneHotEncoder(sparse_output=False, drop="first")
# encoded_categorical = encoder.fit_transform(features[categorical_columns])

# # Combine encoded categorical variables with numerical ones
# encoded_features = np.hstack((encoded_categorical, features[numerical_columns].values))

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     encoded_features, target, test_size=0.2, random_state=42
# )

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predict and calculate residuals on the test set
# y_pred = model.predict(X_test)
# residuals = np.abs(y_test - y_pred)

# # Shift y-values to make all values positive
# shift_value = abs(min(y_test.min(), y_pred.min(), residuals.min()))
# y_test_shifted = y_test + shift_value
# y_pred_shifted = y_pred + shift_value
# residuals_shifted = residuals + shift_value

# # Normalize residuals to derive certainty values (1 - normalized residuals)
# certainty = 1 - (residuals_shifted / residuals_shifted.max())
# certainty_values = certainty.values  # Ensure we have values for visualization

# # Parameters for heatmap visualization
# num_rows = 100
# time_delay = 0.2
# batch_size = 10  # Steps to process per batch
# line_chart_delay = 30  # 2-second delay for line chart

# # Determine the full range of y-axis values (spanning actual and predicted)
# y_min = min(y_test_shifted.min(), y_pred_shifted.min())
# y_max = max(y_test_shifted.max(), y_pred_shifted.max())
# y_range = np.linspace(y_min, y_max, num_rows)


# # Generate heatmap matrix based on prediction range
# def generate_certainty_heatmap(y_pred_value, certainty_value):
#     """
#     Generate a single column of certainty heatmap based on the prediction range and certainty value.
#     """
#     column = 1 - np.abs(y_range - y_pred_value) / (
#         y_max - y_min
#     )  # Intensity based on proximity
#     column = column * certainty_value  # Scale by certainty
#     column = np.clip(column, 0, 1)  # Normalize to [0, 1]
#     return column


# # Initialize heatmap matrix
# columns = len(y_pred_shifted)
# mat = np.zeros((num_rows, columns))

# dashboard = lc.Dashboard(rows=2, columns=1, theme=lc.Themes.Dark)

# # ---- First Row: Real y_pred and y_test Visualization ----
# chart1 = dashboard.ChartXY(row_index=0, column_index=0).set_title(
#     "Actual vs Predicted Values (Unnormalized)"
# )

# legend = chart1.add_legend().set_dragging_mode("draggable")

# # Add line series for actual and predicted values
# actual_series = chart1.add_line_series().set_name("Actual Values")
# actual_series.set_line_color(lc.Color(255, 255, 0)).set_line_thickness(2)

# predicted_series = chart1.add_line_series().set_name("Predicted Values")
# predicted_series.set_line_color(lc.Color(0, 128, 255)).set_line_thickness(2)

# legend.add(actual_series).add(predicted_series)
# # ---- Second Row: Heatmap with Predictions ----
# chart2 = dashboard.ChartXY(row_index=1, column_index=0).set_title(
#     "Real-Time Certainty Heatmap with Line Chart"
# )


# # Add a heatmap grid series
# heatmap_series = chart2.add_heatmap_grid_series(
#     columns=columns, rows=num_rows, data_order="rows"
# ).set_name("Real-Time Certainty Heatmap")
# heatmap_series.set_step(
#     x=(len(y_pred_shifted) - 1) / columns,  # Match the time step
#     y=(y_max - y_min) / num_rows,  # Scale y step to span the prediction range
# )

# # Set the color palette for the heatmap
# heatmap_series.set_palette_coloring(
#     steps=[
#         {
#             "value": 0.0,
#             "color": lc.Color(0, 0, 0, 128),
#         },  # Transparent for low certainty
#         {
#             "value": 0.25,
#             "color": lc.Color(255, 255, 0, 128),
#         },  # Yellow for medium certainty
#         {
#             "value": 0.5,
#             "color": lc.Color(255, 192, 0, 255),
#         },  # Orange for higher certainty
#         {"value": 1.0, "color": lc.Color(255, 0, 0, 255)},  # Red for maximum certainty
#     ],
#     look_up_property="value",
#     percentage_values=True,
#     interpolate=True,
# )

# # Customize heatmap appearance
# heatmap_series.hide_wireframe()
# heatmap_series.set_intensity_interpolation(True)

# # Add actual predictions as a line series
# prediction_series = chart2.add_line_series()
# prediction_series.set_name("Predictions").set_line_color(
#     lc.Color(0, 128, 255)
# ).set_line_thickness(2)

# # Open the chart in live mode
# dashboard.open(live=True, method="browser")

# # Stream certainty values and predictions in batches
# for t in range(0, columns, batch_size):
#     for i in range(batch_size):
#         if t + i < len(y_pred):
#             actual_series.add(x=t + i, y=y_test.iloc[t + i])
#             predicted_series.add(x=t + i, y=y_pred[t + i])
#     # Update heatmap
#     for i in range(batch_size):
#         if t + i < columns:
#             mat[:, t + i] = generate_certainty_heatmap(
#                 y_pred_shifted[t + i], certainty_values[t + i]
#             )

#     # Invalidate the heatmap for the updated batch
#     heatmap_series.invalidate_intensity_values(mat)

#     # Delay the line chart update by 2 seconds (converts delay to steps)
#     time.sleep(time_delay)
#     if t >= int(line_chart_delay / time_delay):
#         for i in range(batch_size):
#             if t + i - int(line_chart_delay / time_delay) < columns:
#                 prediction_series.add(
#                     x=t + i - int(line_chart_delay / time_delay),
#                     y=y_pred_shifted[t + i - int(line_chart_delay / time_delay)],
#                 )
#     # Add remaining line chart points
# for t in range(columns - int(line_chart_delay / time_delay), columns):
#     prediction_series.add(x=t, y=y_pred_shifted[t])

# dashboard.close()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor  # Import XGBoost
import lightningchart as lc
import time

# Load license key
with open(
    "D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt", "r"
) as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

# Load the dataset
data = pd.read_csv("Dataset/mixed_categorical_numerical_data.csv")

# Separate the target variable (Y) and features (X)
target = data["Y"]
features = data.drop(columns=["Y"])

# Identify categorical and numerical columns
categorical_columns = ["X1", "X2", "X3", "X4", "X5"]
numerical_columns = [col for col in features.columns if col not in categorical_columns]

# One-hot encode the categorical variables
encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_categorical = encoder.fit_transform(features[categorical_columns])

# Combine encoded categorical variables with numerical ones
encoded_features = np.hstack((encoded_categorical, features[numerical_columns].values))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    encoded_features, target, test_size=0.2, random_state=42
)

# Train an XGBoost regression model
model = XGBRegressor(objective="reg:squarederror", random_state=42)
model.fit(X_train, y_train)

# Predict and calculate residuals on the test set
y_pred = model.predict(X_test)
residuals = np.abs(y_test - y_pred)

# Shift y-values to make all values positive
shift_value = abs(min(y_test.min(), y_pred.min(), residuals.min()))
y_test_shifted = y_test + shift_value
y_pred_shifted = y_pred + shift_value
residuals_shifted = residuals + shift_value

# Normalize residuals to derive certainty values (1 - normalized residuals)
certainty = 1 - (residuals_shifted / residuals_shifted.max())
certainty_values = certainty.values  # Ensure we have values for visualization

# Parameters for heatmap visualization
num_rows = 100
time_delay = 0.2
batch_size = 10  # Steps to process per batch
line_chart_delay = 30  # 2-second delay for line chart

# Determine the full range of y-axis values (spanning actual and predicted)
y_min = min(y_test_shifted.min(), y_pred_shifted.min())
y_max = max(y_test_shifted.max(), y_pred_shifted.max())
y_range = np.linspace(y_min, y_max, num_rows)


# Generate heatmap matrix based on prediction range
def generate_certainty_heatmap(y_pred_value, certainty_value):
    """
    Generate a single column of certainty heatmap based on the prediction range and certainty value.
    """
    column = 1 - np.abs(y_range - y_pred_value) / (
        y_max - y_min
    )  # Intensity based on proximity
    column = column * certainty_value  # Scale by certainty
    column = np.clip(column, 0, 1)  # Normalize to [0, 1]
    return column


# Initialize heatmap matrix
columns = len(y_pred_shifted)
mat = np.zeros((num_rows, columns))

dashboard = lc.Dashboard(rows=2, columns=1, theme=lc.Themes.Dark)

# ---- First Row: Real y_pred and y_test Visualization ----
chart1 = dashboard.ChartXY(row_index=0, column_index=0).set_title(
    "Actual vs Predicted Values (Unnormalized)"
)

legend = chart1.add_legend().set_dragging_mode("draggable")

# Add line series for actual and predicted values
actual_series = chart1.add_line_series().set_name("Actual Values")
actual_series.set_line_color(lc.Color(255, 255, 0)).set_line_thickness(2)

predicted_series = chart1.add_line_series().set_name("Predicted Values")
predicted_series.set_line_color(lc.Color(0, 128, 255)).set_line_thickness(2)

legend.add(actual_series).add(predicted_series)
# ---- Second Row: Heatmap with Predictions ----
chart2 = dashboard.ChartXY(row_index=1, column_index=0).set_title(
    "Real-Time Certainty Heatmap with Line Chart"
)


# Add a heatmap grid series
heatmap_series = chart2.add_heatmap_grid_series(
    columns=columns, rows=num_rows, data_order="rows"
).set_name("Real-Time Certainty Heatmap")
heatmap_series.set_step(
    x=(len(y_pred_shifted) - 1) / columns,  # Match the time step
    y=(y_max - y_min) / num_rows,  # Scale y step to span the prediction range
)

# Set the color palette for the heatmap
heatmap_series.set_palette_coloring(
    steps=[
        {
            "value": 0.0,
            "color": lc.Color(0, 0, 0, 128),
        },  # Transparent for low certainty
        {
            "value": 0.25,
            "color": lc.Color(255, 255, 0, 128),
        },  # Yellow for medium certainty
        {
            "value": 0.5,
            "color": lc.Color(255, 192, 0, 255),
        },  # Orange for higher certainty
        {"value": 1.0, "color": lc.Color(255, 0, 0, 255)},  # Red for maximum certainty
    ],
    look_up_property="value",
    percentage_values=True,
    interpolate=True,
)

# Customize heatmap appearance
heatmap_series.hide_wireframe()
heatmap_series.set_intensity_interpolation(True)

# Add actual predictions as a line series
prediction_series = chart2.add_line_series()
prediction_series.set_name("Predictions").set_line_color(
    lc.Color(0, 128, 255)
).set_line_thickness(2)

# Open the chart in live mode
dashboard.open(live=True, method="browser")

# Stream certainty values and predictions in batches
for t in range(0, columns, batch_size):
    for i in range(batch_size):
        if t + i < len(y_pred):
            actual_series.add(x=t + i, y=float(y_test.iloc[t + i]))
            predicted_series.add(x=t + i, y=float(y_pred[t + i]))
    # Update heatmap
    for i in range(batch_size):
        if t + i < columns:
            mat[:, t + i] = generate_certainty_heatmap(
                y_pred_shifted[t + i], certainty_values[t + i]
            )

    # Invalidate the heatmap for the updated batch
    heatmap_series.invalidate_intensity_values(mat)

    # Delay the line chart update by 2 seconds (converts delay to steps)
    time.sleep(time_delay)
    if t >= int(line_chart_delay / time_delay):
        for i in range(batch_size):
            if t + i - int(line_chart_delay / time_delay) < columns:
                prediction_series.add(
                    x=t + i - int(line_chart_delay / time_delay),
                    y=float(y_pred_shifted[t + i - int(line_chart_delay / time_delay)]),
                )
# Add remaining line chart points
for t in range(columns - int(line_chart_delay / time_delay), columns):
    prediction_series.add(x=t, y=float(y_pred_shifted[t]))


dashboard.close()
