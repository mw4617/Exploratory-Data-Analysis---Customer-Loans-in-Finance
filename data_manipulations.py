import pandas as pd
from scipy.stats import normaltest
import matplotlib.pyplot as plt  # Add this line to import matplotlib.pyplot
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import boxcox


class DataTransform:

    def __init__(self,eda_data_frame):

        self.eda_data_frame=eda_data_frame

    def column_conversions(self):

        # Convert 'term' column to integers by extracting digits
        self.eda_data_frame['term'] = self.eda_data_frame['term'].str.extract('(\d+)').astype('Int64')    

        #definining a function for employment length conversion
        def employment_length_conversion(length):
            
            if pd.isna(length):  # Check for NaN or missing values
                
                return None
            
            elif isinstance(length, str):  # Ensure the value is a string
                
                if '<1 year' in length:
                    
                    return 0
                    
                elif '1 year' in length:
                        
                    return 1
                
                elif '2 years' in length:
                    
                    return 2

                elif '3 years' in length:

                    return 3

                elif '4 years' in length:
            
                    return 4

                elif '5 years' in length:
                
                    return 5
            
                elif '6 years' in length:
                
                    return 6
            
                elif '7 years' in length:
                    
                    return 7
            
                elif '8 years' in length:
                
                    return 8
            
                elif '9 years' in length:
                
                    return 9
            
                elif '10+ years' in length:
                
                    return 10

        self.eda_data_frame['employment_length'] = self.eda_data_frame['employment_length'].apply(employment_length_conversion)

        return self.eda_data_frame

class DataFrameInfo:

    def __init__(self,eda_data_frame):

        self.eda_data_frame=eda_data_frame

    def column_statistics(self):
        # Total number of rows
        no_rows = self.eda_data_frame.shape[0]
        
        # Iterate through each column in the DataFrame
        for col in self.eda_data_frame.columns:
            print(f"\nColumn: {col}")

            # Calculate missing values percentage
            missing_value_count = no_rows - self.eda_data_frame[col].count()
            missing_percentage = (missing_value_count / no_rows) * 100
            print(f"Missing Percentage: {missing_percentage:.2f}%, Missing Value Count: {missing_value_count}")

            # Perform normality test for numeric columns (D'Agostino's K^2 Test)
            if pd.api.types.is_numeric_dtype(self.eda_data_frame[col]):
                variable_data = self.eda_data_frame[col]
                stat, p = normaltest(variable_data, nan_policy='omit')
                print(f"Normality Test - Statistics={stat:.3f}, p={p:.3f}")
                
                if p > 0.05:
                    print("There is insufficient evidence to suggest that data does not follow normal distribution (fail to reject H0).")
                else:
                    print("The data does not appear to follow a normal distribution (reject H0).")
            else:
                print("Normality Test: Not applicable (non-numeric column).")



    
class Plotter:
    
    def __init__(self,eda_data_frame):

        self.eda_data_frame=eda_data_frame

    def histogram(self):

        # Iterate through each column in the DataFrame
        for col in self.eda_data_frame.columns:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(self.eda_data_frame[col]):
                plt.figure(figsize=(8, 6))
                plt.hist(self.eda_data_frame[col].dropna(), bins=1000, edgecolor='black', alpha=0.7)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.show()

                print(f'The skew of {col} column is {self.eda_data_frame[col].skew()}')
            else:
                print(f"Skipping non-numeric column: {col}")

    def q_q_plot(self):

        # Iterate through each column in the DataFrame
        for col in self.eda_data_frame.columns:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(self.eda_data_frame[col]):
                plt.figure(figsize=(8, 6))
                qqplot(self.eda_data_frame[col].dropna(), scale=1 ,line='q')
                plt.title(f'Q-Q Plot of {col}')
                plt.grid(axis='both', alpha=0.75)
                plt.show()
            else:
                print(f"Skipping non-numeric column: {col}")


    def plot_null_values_heatmap(self, after_df):
        """
        Plots a heatmap to compare null values before and after cleanup,
        including columns that are only in the DataFrame before cleanup but not after cleanup.
        
        Parameters:
            after_df (pd.DataFrame): DataFrame after cleanup.
        """
        # Calculate null values for each column in both DataFrames
        before_nulls = self.eda_data_frame.isnull().sum()
        after_nulls = after_df.isnull().sum()

        # Reindex after_nulls to include columns from before_df, filling missing columns with NaN
        all_columns = before_nulls.index
        after_nulls = after_nulls.reindex(all_columns, fill_value=float('NaN'))

        # Create a DataFrame for the heatmap
        heatmap_data = pd.DataFrame({
            'Before Cleanup': before_nulls,
            'After Cleanup': after_nulls
        }).reset_index()
        heatmap_data = heatmap_data.melt(id_vars='index', var_name='Cleanup Stage', value_name='Null Values')
        heatmap_data.rename(columns={'index': 'Column'}, inplace=True)

        # Reorder Cleanup Stage so "Before Cleanup" comes first
        cleanup_order = ['Before Cleanup', 'After Cleanup']
        heatmap_data['Cleanup Stage'] = pd.Categorical(heatmap_data['Cleanup Stage'], categories=cleanup_order, ordered=True)

        # Apply cube root transformation for the heatmap's color representation
        heatmap_data['Null Values (Transformed)'] = heatmap_data['Null Values'].apply(lambda x: np.cbrt(x) if not np.isnan(x) else np.nan)

        # Create the heatmap
        plt.figure(figsize=(10, len(all_columns) * 0.25))  # Dynamically adjust height for readability
        pivot_data = heatmap_data.pivot(index='Column', columns='Cleanup Stage', values='Null Values (Transformed)')
        
        # Set up the heatmap with annotations and a custom legend
        ax = sns.heatmap(
            pivot_data,
            annot=heatmap_data.pivot(index='Column', columns='Cleanup Stage', values='Null Values'),
            fmt='.0f',
            cmap='coolwarm',
            cbar_kws={'label': 'Number of Null Values'}
        )
        
        # Update colorbar for visually equidistant points
        colorbar = ax.collections[0].colorbar
        max_value = heatmap_data['Null Values'].max()

        # Generate equidistant ticks in the cube root scale
        num_ticks = 6  # Define the number of ticks
        cube_root_ticks = np.linspace(0, np.cbrt(max_value), num_ticks)  # Evenly spaced in cube root scale
        original_values = [int(round(t**3)) for t in cube_root_ticks]  # Convert back to original values for labels
        colorbar.set_ticks(cube_root_ticks)
        colorbar.set_ticklabels(original_values)

        # Finalize plot
        plt.title('Number of Null Values Before and After Cleanup')
        plt.xlabel('Cleanup Stage')
        plt.ylabel('Column Name')
        plt.tight_layout()
        plt.show()

    def compare_transformations(self, transformed_dataframes):
        """
        Plots comparison graphs for 9 transformations across numerical columns.

        Args:
            transformed_dataframes (list): A list of 9 DataFrames, each representing
                                           a transformation method.

        For each numerical column:
            - Graph 1: A 3x3 grid of histograms for each transformation, overlaid with
                       a perfect normal distribution curve (red line).
            - Graph 2: A 3x3 grid of Q-Q plots for each transformation.
        """
        transformation_names = [
            "Log",
            "Cube Root",
            "Arcsinh",
            "Quantile",
            "Custom Power 1",
            "Custom Power 2",
            "Box-Cox",
            "Yeo-Johnson",
            "Custom Factorial"
        ]

        for col in self.eda_data_frame.select_dtypes(include='number').columns:
            print(f"Comparing transformations for column: {col}")

            # Create a 3x3 grid of histograms
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle(f'Histograms of Transformations for {col}', fontsize=16)

            for i, ax in enumerate(axes.flatten()):
                if i < len(transformed_dataframes):
                    df = transformed_dataframes[i]
                    data = df[col].dropna()
                    
                    # Plot histogram
                    ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black', label=transformation_names[i])

                    # Overlay normal distribution
                    mu, sigma = norm.fit(data)
                    x = np.linspace(data.min(), data.max(), 1000)
                    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', label='Normal Distribution')

                    ax.set_title(transformation_names[i])
                    ax.legend()
                    ax.grid(axis='y', alpha=0.75)
                else:
                    ax.axis('off')  # Turn off unused subplots

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

            # Create a 3x3 grid of Q-Q plots
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle(f'Q-Q Plots of Transformations for {col}', fontsize=16)

            for i, ax in enumerate(axes.flatten()):
                if i < len(transformed_dataframes):
                    df = transformed_dataframes[i]
                    data = df[col].dropna()

                    # Q-Q plot
                    probplot(data, dist="norm", plot=ax)
                    ax.get_lines()[1].set_color('r')  # Red line for theoretical quantiles
                    ax.set_title(transformation_names[i])
                    ax.grid(alpha=0.75)
                else:
                    ax.axis('off')  # Turn off unused subplots

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

class DataFrameTransform:

    def __init__(self,eda_data_frame):
        """
        Initialize the DataFrameTransform class.

        Args:
            eda_data_frame (pd.DataFrame): The input DataFrame to be transformed.
        """
        self.eda_data_frame=eda_data_frame

        self.no_rows=len(self.eda_data_frame)

        self.transformed_dataframes = {}  # To store the 10 transformed DataFrames

    
    def impute_missing_values(self, threshold=20, strategy='mean'):
        """
        Impute missing values in a DataFrame for each column with the mean or median
        if the percentage of missing values is less than the threshold percentage.
        
        Parameters:
            threshold (float): Maximum percentage of missing values allowed for imputation (20% default).
            strategy (str): Imputation strategy, either 'mean' or 'median' (mean default).
            
        Returns:
           self.eda_data_frame (pd.DataFrame): DataFrame with missing values imputed.
        """
        for col in self.eda_data_frame.columns:
            if pd.api.types.is_numeric_dtype(self.eda_data_frame[col]):  # Apply only to numeric columns
                missing_percentage = self.eda_data_frame[col].isnull().sum() / self.no_rows * 100
                
                if missing_percentage < threshold:
                    if strategy == 'mean':
                        replacement_value = self.eda_data_frame[col].mean()
                    elif strategy == 'median':
                        replacement_value = self.eda_data_frame[col].median()
                    else:
                        raise ValueError("Invalid strategy. Use 'mean' or 'median'.")
                    
                    self.eda_data_frame[col].fillna(replacement_value, inplace=True)
                    print(f"Imputed missing values in column '{col}' with its {strategy}.")
                else:
                    print(f"Skipped column '{col}' (missing percentage: {missing_percentage:.2f}%).")
            else:
                print(f"Skipped non-numeric column: '{col}'.")

        return self.eda_data_frame    


def drop_columns_with_missing_and_constant_values(self, threshold=50):
    """
    Drops columns from the DataFrame if:
    - The percentage of missing values equals or exceeds the given threshold, OR
    - All values in the column are the same (constant column).

    Parameters:
        threshold (float): The maximum allowable percentage of missing values (50% by default).

    Returns:
        self.eda_data_frame (pd.DataFrame): A DataFrame with columns dropped based on the threshold or constant values.
    """
    # Calculate the percentage of missing values for each column
    missing_percentage = self.eda_data_frame.isnull().sum() / self.no_rows * 100

    # Identify columns to drop due to missing values
    columns_to_drop = missing_percentage[missing_percentage >= threshold].index.tolist()

    # Identify constant columns (all values are the same)
    constant_columns = [col for col in self.eda_data_frame.columns if self.eda_data_frame[col].nunique() == 1]

    # Combine both sets of columns to drop
    columns_to_drop.extend(constant_columns)

    # Drop the columns from the DataFrame
    self.eda_data_frame = self.eda_data_frame.drop(columns=columns_to_drop)

    # Print information about dropped columns
    if columns_to_drop:
        print(f"Dropped columns: {columns_to_drop}")
    else:
        print("No columns were dropped.")

    return self.eda_data_frame

    def drop_rows_with_nulls_in_columns(self, columns):
        """
        Drops rows from the DataFrame if any of the specified columns have null values.

        Parameters:
            columns (list): List of column names to check for null values.

        Returns:
           self.eda_data_frame (pd.DataFrame): A DataFrame with rows dropped based on null values.
        """
        # Ensure the specified columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in self.eda_data_frame.columns]

        if missing_columns:

            raise ValueError(f"The following columns do not exist in the DataFrame: {missing_columns}")

        # Drop rows with null values in the specified columns

        before_rows = self.eda_data_frame.shape[0]

        self.eda_data_frame = self.eda_data_frame.dropna(subset=columns)

        after_rows = self.eda_data_frame.shape[0]

        # Display information about the rows dropped
        rows_dropped = before_rows - after_rows

        print(f"Dropped {rows_dropped} rows due to null values in columns: {columns}")

        return self.eda_data_frame        

    def log_transformation(self):
        """
        Applies a log transformation to all numerical columns in the DataFrame.
        Adjusts values to ensure all are positive by adding abs(min) + 1 if necessary.
        Stores the result in the `transformed_dataframes` dictionary under the key 'log'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            min_value = df_transformed[col].min()

            if min_value <= 0:

                adjustment = abs(min_value) + 1

                df_transformed[col] += adjustment

            df_transformed[col] = df_transformed[col].apply(lambda x: np.log(x) if pd.notnull(x) else x)

        self.transformed_dataframes['log'] = df_transformed

    def cube_root_transformation(self):
        """
        Applies a cube root transformation to all numerical columns in the DataFrame.
        Stores the result in the `transformed_dataframes` dictionary under the key 'cube_root'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            df_transformed[col] = df_transformed[col].apply(lambda x: x**(1/3) if pd.notnull(x) else x)

        self.transformed_dataframes['cube_root'] = df_transformed

    def arcsinh_transformation(self):
        """
        Applies an arcsinh transformation to all numerical columns in the DataFrame.
        The arcsinh transformation is similar to the log transformation but handles negative values natively.
        Stores the result in the `transformed_dataframes` dictionary under the key 'arcsinh'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            df_transformed[col] = df_transformed[col].apply(lambda x: np.arcsinh(x) if pd.notnull(x) else x)

        self.transformed_dataframes['arcsinh'] = df_transformed

    def quantile_transformation(self, output_distribution='uniform'):
        """
        Applies a quantile transformation to all numerical columns in the DataFrame.
        Maps data to follow a uniform or normal distribution.

        Args:
            output_distribution (str): The target distribution for the transformation ('uniform' or 'normal').

        Stores the result in the `transformed_dataframes` dictionary under the key 'quantile'.
        """
        df_transformed = self.eda_data_frame.copy()

        transformer = QuantileTransformer(output_distribution=output_distribution, random_state=42)

        for col in df_transformed.select_dtypes(include='number').columns:

            reshaped_col = df_transformed[col].values.reshape(-1, 1)

            df_transformed[col] = transformer.fit_transform(reshaped_col)

        self.transformed_dataframes['quantile'] = df_transformed

    def custom_power_transformation_1(self):
        """
        Applies a custom power transformation defined as x^(1 / (x^(6/5))) to all numerical columns in the DataFrame.
        Handles only positive values; for others, NaN is returned.
        Stores the result in the `transformed_dataframes` dictionary under the key 'custom_power_1'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            df_transformed[col] = df_transformed[col].apply(lambda x: x**(1 / (x**(6/5))) if pd.notnull(x) and x > 0 else x)

        self.transformed_dataframes['custom_power_1'] = df_transformed

    def custom_power_transformation_2(self):
        """
        Applies a custom power transformation defined as x^(1 / (x^2)) to all numerical columns in the DataFrame.
        Handles only positive values; for others, NaN is returned.
        Stores the result in the `transformed_dataframes` dictionary under the key 'custom_power_2'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            df_transformed[col] = df_transformed[col].apply(lambda x: x**(1 / (x**2)) if pd.notnull(x) and x > 0 else x)

        self.transformed_dataframes['custom_power_2'] = df_transformed

    def box_cox_transformation(self):
        """
        Applies a Box-Cox transformation to all numerical columns in the DataFrame.
        Adjusts values to ensure all are positive by adding abs(min) + 1 if necessary.
        Stores the result in the `transformed_dataframes` dictionary under the key 'box_cox'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            min_value = df_transformed[col].min()

            if min_value <= 0:

                adjustment = abs(min_value) + 1

                df_transformed[col] += adjustment

            df_transformed[col] = boxcox(df_transformed[col].dropna())[0]

        self.transformed_dataframes['box_cox'] = df_transformed

    def yeo_johnson_transformation(self):
        """
        Applies a Yeo-Johnson transformation to all numerical columns in the DataFrame.
        The Yeo-Johnson transformation can handle both positive and negative values.
        Stores the result in the `transformed_dataframes` dictionary under the key 'yeo_johnson'.
        """
        df_transformed = self.eda_data_frame.copy()

        transformer = PowerTransformer(method='yeo-johnson', standardize=False)

        for col in df_transformed.select_dtypes(include='number').columns:

            reshaped_col = df_transformed[col].values.reshape(-1, 1)

            df_transformed[col] = transformer.fit_transform(reshaped_col)

        self.transformed_dataframes['yeo_johnson'] = df_transformed

    def custom_transform_factorial(self):
        """
        Applies a custom transformation defined as (x^(0.8x)) / factorial(round(x)) to all numerical columns.
        Handles only non-negative values; for others, NaN is returned.
        Stores the result in the `transformed_dataframes` dictionary under the key 'custom_factorial'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            def transform(x):

                if pd.notnull(x) and x >= 0:

                    rounded_x = round(x)

                    try:
                        return (x**(0.8 * x)) / math.factorial(rounded_x)

                    except (OverflowError, ValueError):

                        return np.nan

                return np.nan

            df_transformed[col] = df_transformed[col].apply(transform)

        self.transformed_dataframes['custom_factorial'] = df_transformed

    def apply_all_transforms(self):
        """
        Applies all 9 transformations to the DataFrame.

        The transformations include:
            - Log Transformation
            - Cube Root Transformation
            - Arcsinh Transformation
            - Quantile Transformation
            - Custom Power Transformation 1
            - Custom Power Transformation 2
            - Box-Cox Transformation
            - Yeo-Johnson Transformation
            - Custom Factorial-Based Transformation

        Returns:
            list: A list of transformed DataFrames.
        """
        self.log_transformation()

        self.cube_root_transformation()

        self.arcsinh_transformation()

        self.quantile_transformation()

        self.custom_power_transformation_1()

        self.custom_power_transformation_2()

        self.box_cox_transformation()

        self.yeo_johnson_transformation()

        self.custom_transform_factorial()

        return [self.transformed_dataframes[key] for key in self.transformed_dataframes.keys()]
