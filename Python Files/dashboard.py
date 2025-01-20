import pandas as pd
from sklearn.decomposition import PCA
import lightningchart as lc

with open(
    "D:/Computer Aplication/WorkPlacement/Projects/shared_variable.txt", "r"
) as f:
    mylicensekey = f.read().strip()
lc.set_license(mylicensekey)

data = pd.read_csv("Dataset/mixed_categorical_numerical_data.csv")

# Create a dashboard with 2 rows and 2 columns
dashboard = lc.Dashboard(rows=2, columns=2, theme=lc.Themes.Light)

####################### chart 1 ######################
x_categories = ["X1", "X2", "X3", "X4", "X5"]  # Features for x-axis
sub_categories = []  # To hold unique types for all features
stacked_data = {}

# Get unique subcategories for each feature and compute their counts
for feature in x_categories:
    unique_values = sorted(data[feature].unique())  # Get unique types
    sub_categories.extend(unique_values)  # Append unique types to sub_categories
    for value in unique_values:
        if value not in stacked_data:
            stacked_data[value] = [0] * len(x_categories)

    # Count occurrences for the feature
    feature_counts = data[feature].value_counts()
    for value, count in feature_counts.items():
        feature_index = x_categories.index(feature)
        stacked_data[value][feature_index] = count

# Prepare data for the chart
stacked_bar_data = [
    {"subCategory": sub_category, "values": stacked_data[sub_category]}
    for sub_category in sub_categories
    if sub_category in stacked_data
]

# Create a Stacked Bar Chart
chart1 = (
    dashboard.BarChart(row_index=0, column_index=0)
    .set_title("Count of Different Machine and Product Types by Feature (X1 to X5)")
    .set_title_font(size=18, weight="bold")
)

# Set the stacked bar data
chart1.set_data_stacked(
    x_categories,  # X1 to X5 as the x-axis labels
    stacked_bar_data,  # Subcategories and their counts
)

# Customize the chart
chart1.set_value_label_display_mode("hidden")

####################### chart 2 ######################
numeric_data = data.drop(columns=["X1", "X2", "X3", "X4", "X5", "Y"])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(numeric_data)

# Extract PCA components and throughput rate (Y)
pca_x = pca_result[:, 0]
pca_y = pca_result[:, 1]
throughput_rate = data["Y"]

# Create a LightningChart scatter plot
chart2 = (
    dashboard.ChartXY(row_index=1, column_index=0)
    .set_title("PCA Visualization of Numeric Data")
    .set_title_font(size=18, weight="bold")
)

# Add a point series with palette-based coloring
point_series = chart2.add_point_series(colors=True, lookup_values=True)

# Define a palette for coloring
point_series.set_palette_point_coloring(
    steps=[
        {"value": throughput_rate.min(), "color": lc.Color(0, 0, 255)},  # Blue for min
        {"value": throughput_rate.max(), "color": lc.Color(255, 0, 0)},  # Red for max
    ],
    look_up_property="value",  # Use 'value' for coloring
    interpolate=True,
)

# Append each point to the series
for x, y, value in zip(pca_x, pca_y, throughput_rate):
    point_series.append_sample(x=x, y=y, lookup_value=value)

# Set axis labels
chart2.get_default_x_axis().set_title("Principal Component 1")
chart2.get_default_y_axis().set_title("Principal Component 2")

chart2.add_legend(data=point_series, title="Throughput Rate (Y)")

############ chart 3 #######################
observation_order = data.index
throughput_rate = data["Y"]

# Create a LightningChart Line Chart
chart3 = (
    dashboard.ChartXY(row_index=0, column_index=1)
    .set_title("Throughput Rate Trends Over Observation Order")
    .set_title_font(size=18, weight="bold")
)

# Add a Line Series
line_series = chart3.add_line_series()
line_series.append_samples(x_values=observation_order, y_values=throughput_rate)
line_series.set_line_color(lc.Color("blue")).set_line_thickness(2)

# Customize Axes
chart3.get_default_x_axis().set_title("Observation Order")
chart3.get_default_y_axis().set_title("Throughput Rate (Y)")

############### chart 4 ######################

# Extract necessary columns
x6 = data["X6"].values
x7 = data["X7"].values
y = data["Y"].values  # Used for both size and color of the bubbles

# Create the chart
chart4 = (
    dashboard.ChartXY(row_index=1, column_index=1)
    .set_title("Bubble Chart: X6 vs. X7 (Bubble Size = Y)")
    .set_title_font(size=18, weight="bold")
)

# Add a bubble series
bubble_series = chart4.add_point_series(
    sizes=True,
    lookup_values=True,  # For coloring based on `Y`
)
y_max = max(y)
# Add data to the bubble chart
bubble_series.append_samples(
    x_values=x6,
    y_values=x7,
    sizes=y / y_max * 50,  # Bubble size based on Y
    lookup_values=y,  # Bubble color based on Y
)

# Enable individual point coloring and set a color palette
bubble_series.set_individual_point_color_enabled(True)
bubble_series.set_palette_point_coloring(
    steps=[
        {"value": y.min(), "color": lc.Color(0, 0, 255)},  # Blue for low Y values
        {"value": y.max(), "color": lc.Color(255, 0, 0)},  # Red for high Y values
    ],
    look_up_property="value",
    interpolate=True,
)
chart4.add_legend(data=bubble_series, title="Throughput Rate (Y)")
# Configure the chart axes
chart4.get_default_x_axis().set_title("Grinding Thickness (X6)")
chart4.get_default_y_axis().set_title("Feature (X7)")

# Open the dashboard
dashboard.open(method="browser")
