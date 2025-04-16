import pandas as pd
from sklearn.preprocessing import PowerTransformer
import math
import plotly.express as px
from scipy.stats import normaltest, norm, probplot, skew
import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
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


class LoanAnalyzer:
    """
    Analyzes the current repayment status of loans and provides repayment projections.
    """

    def __init__(self, df):
        """
        Initializes the LoanAnalyzer and computes total_paid if it doesn't exist.
        """
        self.df = df.copy()

        if 'total_paid' not in self.df.columns:
            self.df['total_paid'] = (
                self.df['total_rec_prncp'].fillna(0) +
                self.df['total_rec_int'].fillna(0) +
                self.df['total_rec_late_fee'].fillna(0) +
                self.df['recoveries'].fillna(0)
            )

    def calculate_recovery_stats(self):
        """
        Calculates total expected payment, total paid to date, and the percentage recovered.
        Also prints diagnostics to verify amounts.
        """
        total_expected = self.df['total_payment'].sum()
        total_paid = self.df['total_paid'].sum()
        recovery_percentage = (total_paid / total_expected) * 100 if total_expected else 0

        # Diagnostic print block
        print(f"Total expected payment (sum): ${total_expected:,.2f}")
        print(f"Total principal recovered:    ${self.df['total_rec_prncp'].sum():,.2f}")
        print(f"Total interest recovered:     ${self.df['total_rec_int'].sum():,.2f}")
        print(f"Total late fees recovered:    ${self.df['total_rec_late_fee'].sum():,.2f}")
        print(f"Recoveries:                   ${self.df['recoveries'].sum():,.2f}")
        print(f"Total paid (sum):             ${total_paid:,.2f}")
        print(f"Remaining to be paid:         ${total_expected - total_paid:,.2f}")

        return {
            'total_expected': total_expected,
            'total_paid': total_paid,
            'recovery_percentage': recovery_percentage
        }

    def calculate_next_6_months_projection(self):
        """
        Projects the total repayment expected in the next 6 months.
        It ensures no loan is projected to pay more than its remaining balance,
        and the total projection does not exceed the actual remaining balance.
        """
        # Recompute total_paid and remaining balances
        self.df['total_paid'] = (
            self.df['total_rec_prncp'].fillna(0) +
            self.df['total_rec_int'].fillna(0) +
            self.df['total_rec_late_fee'].fillna(0) +
            self.df['recoveries'].fillna(0)
        )

        # Global expected remaining (same as in diagnostic print)
        total_expected = self.df['total_payment'].sum()
        total_paid = self.df['total_paid'].sum()
        total_remaining = max(total_expected - total_paid, 0)  # global cap

        # Per-loan level projections (min of 6*instalment vs remaining)
        self.df['remaining'] = (self.df['total_payment'] - self.df['total_paid']).clip(lower=0)
        self.df['projected_payment'] = self.df['instalment'].fillna(0) * 6
        self.df['projected_payment'] = self.df[['projected_payment', 'remaining']].min(axis=1)

        # Final capped result
        projected_sum = self.df['projected_payment'].sum()
        return min(round(projected_sum, 2), round(total_remaining, 2))


    def get_repayment_summary(self):
        """
        Combines all repayment analysis into a single dictionary:
        - Total expected
        - Total paid
        - Recovery percentage
        - Next 6 months projection
        """
        stats = self.calculate_recovery_stats()
        stats['next_6_months_projection'] = self.calculate_next_6_months_projection()
        return stats
    
    def get_charged_off_loss_summary(self):
        """
        Calculates and summarizes losses from charged-off loans.
        Returns the number and percentage of charged-off loans, total loss, and average loss.
        """
        charged_off_df = self.df[self.df['loan_status'] == 'Charged Off'].copy()

        charged_off_df['loss'] = charged_off_df['funded_amount'] - charged_off_df['total_paid']
        charged_off_df['loss'] = charged_off_df['loss'].clip(lower=0)

        total_loans = len(self.df)
        charged_off_loans = len(charged_off_df)

        loss_summary = {
            'total_charged_off': charged_off_loans,
            'charged_off_percentage': (charged_off_loans / total_loans) * 100 if total_loans else 0,
            'total_loss': charged_off_df['loss'].sum(),
            'avg_loss': charged_off_df['loss'].mean() if charged_off_loans else 0
        }

        return loss_summary


    def calculate_projected_loss(self):
        """
        Estimates the projected loss if all charged-off loans had been fully repaid.
        
        Returns:
            dict: Contains actual loss, projected full repayment, and the gap.
        """
        charged_off = self.df[self.df['loan_status'] == 'Charged Off'].copy()

        if charged_off.empty:
            return {'actual_loss': 0, 'projected_loss': 0, 'total_gap': 0}

        # Actual loss: funded amount - total recovered
        charged_off['total_paid'] = (
            charged_off['total_rec_prncp'].fillna(0) +
            charged_off['total_rec_int'].fillna(0) +
            charged_off['total_rec_late_fee'].fillna(0) +
            charged_off['recoveries'].fillna(0)
        )
        charged_off['actual_loss'] = charged_off['funded_amount'] - charged_off['total_paid']
        charged_off['actual_loss'] = charged_off['actual_loss'].clip(lower=0)

        # Projected loss = funded amount - total expected payment
        charged_off['projected_loss'] = charged_off['funded_amount'] - charged_off['total_payment']
        charged_off['projected_loss'] = charged_off['projected_loss'].clip(lower=0)

        total_actual_loss = charged_off['actual_loss'].sum()
        total_projected_loss = charged_off['projected_loss'].sum()
        total_gap = total_actual_loss - total_projected_loss

        return {
            'actual_loss': total_actual_loss,
            'projected_loss': total_projected_loss,
            'total_gap': total_gap
        }

    def get_late_loan_loss_estimate(self):
        """
        Estimates potential losses from late loans by checking shortfall vs funded amount.
        Returns a DataFrame with estimated loss per late loan.
        """
        late_loans = self.df[self.df['loan_status'].str.contains('Late', na=False)].copy()
        late_loans['estimated_remaining'] = late_loans['total_payment'] - late_loans['total_paid']
        late_loans['estimated_loss'] = late_loans['funded_amount'] - late_loans['total_paid'] - late_loans['estimated_remaining']
        late_loans['estimated_loss'] = late_loans['estimated_loss'].clip(lower=0)

        return late_loans[['id', 'loan_status', 'funded_amount', 'total_paid', 'estimated_loss']]


    def tag_loss_risk_users(self, loss_threshold=0.5):
        """
        Tags loans as high loss risk if estimated_loss / funded_amount exceeds threshold.
        Adds a 'high_loss_risk' boolean column.
        """
        late_loans = self.get_late_loan_loss_estimate()
        late_loans['loss_ratio'] = late_loans['estimated_loss'] / late_loans['funded_amount']
        late_loans['high_loss_risk'] = late_loans['loss_ratio'] > loss_threshold

        result = self.df.copy()
        result['high_loss_risk'] = result['id'].isin(late_loans[late_loans['high_loss_risk']]['id'])

        return result

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

    def plot_correlation_heat_map(self):
        """
        Plots a heatmap of the correlation matrix for numeric columns only.
        Uses a cleaner layout and 'viridis' colormap.
        """
        numeric_df = self.eda_data_frame.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = plt.cm.viridis  # New colormap here

        fig, ax = plt.subplots(figsize=(1.2 * len(corr.columns), 1.0 * len(corr.columns)))

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.75},
            ax=ax
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_title('Correlation Matrix of Numerical Features', fontsize=14, pad=20)

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
            "Untransformed",
            "Log",
            "Cube Root",
            "Seventh Root",
            "21st Root",
            "Arcsinh",
            "Arcosh",
            "Box-Cox",
            "Yeo-Johnson"
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

                    # Calculate skewness and format title
                    skew_val = skew(data)
                    title = f"{transformation_names[i]}\nSkewness: {skew_val:.2f}"
                    ax.set_title(title)
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
                    if len(ax.get_lines()) > 1:
                        ax.get_lines()[1].set_color('r')  # Red line for theoretical quantiles
                    ax.set_title(transformation_names[i])
                    ax.grid(alpha=0.75)
                else:
                    ax.axis('off')  # Turn off unused subplots

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

    def plot_loan_repayment_summary(self, total_paid, total_expected, next_6_months_payment):
            """
            Plots a bar chart summarizing loan repayment status:
            - Total paid so far
            - Remaining amount
            - Next 6 months projection

            Parameters:
                total_paid (float): Total amount already paid
                total_expected (float): Total expected payment over loan term
                next_6_months_payment (float): Projected payment in next 6 months
            """
            remaining = max(total_expected - total_paid, 0)

            values = [total_paid, remaining, next_6_months_payment]
            labels = ['Recovered', 'Remaining', 'Next 6 Months']
            colors = ['green', 'gray', 'blue']

            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values, color=colors)

            plt.title("Loan Repayment Summary")
            plt.ylabel("Amount ($, in millions)")
            plt.grid(axis='y', linestyle='--', alpha=0.5)

            # Use formatter to display y-axis in millions
            formatter = ticker.FuncFormatter(lambda x, _: f'${x * 1e-6:,.0f}M')
            plt.gca().yaxis.set_major_formatter(formatter)

            # Add value labels above bars
            for bar in bars:
                yval = bar.get_height()
                # Use full dollar value if less than $1M, else show in millions
                if yval < 1_000_000:
                    label = f"${yval:,.0f}"
                else:
                    label = f"${yval * 1e-6:,.0f}M"

                # Offset label to appear clearly above bar
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + max(total_expected * 0.01, 1e6 * 0.01),  # ~1% of height
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

            plt.tight_layout()
            plt.show()

    def plot_loss_distribution(self):
        """
        Plots a histogram of loss amounts for charged-off loans.
        """
        df = self.eda_data_frame.copy()

        if 'total_paid' not in df.columns:
            df['total_paid'] = (
                df['total_rec_prncp'].fillna(0) +
                df['total_rec_int'].fillna(0) +
                df['total_rec_late_fee'].fillna(0) +
                df['recoveries'].fillna(0)
            )

        if 'loss' not in df.columns:
            df['loss'] = df['funded_amount'] - df['total_paid']

        df['loss'] = df['loss'].clip(lower=0)
        charged_off = df[df['loan_status'] == 'Charged Off']

        plt.figure(figsize=(10, 6))
        plt.hist(charged_off['loss'].dropna(), bins=100, color='red', alpha=0.7, edgecolor='black')
        plt.title('Loss Distribution for Charged-Off Loans')
        plt.xlabel('Loss Amount')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_loss_risk_users(self):
        """
        Plots bar chart of loans tagged as high risk for loss.
        """
        if 'high_loss_risk' not in self.eda_data_frame.columns:
            print("No 'high_loss_risk' column found.")
            return

        risk_counts = self.eda_data_frame['high_loss_risk'].value_counts()
        plt.figure(figsize=(6, 4))
        risk_counts.plot(kind='bar', color=['green', 'red'])
        plt.title('High Loss Risk Tag Distribution')
        plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
        plt.ylabel('Number of Loans')
        plt.tight_layout()
        plt.show()        

    def plot_projected_loss_bar(self, summary):
        """
        Plots a bar chart comparing actual vs projected losses for charged-off loans.
        
        Parameters:
            summary (dict): Dictionary containing:
                - 'actual_loss': Actual loss from charged-off loans
                - 'projected_loss': Estimated loss if charged-off loans had completed repayment
                - 'total_gap': Difference between actual and projected losses
        """
        actual = summary.get('actual_loss', 0)
        projected = summary.get('projected_loss', 0)
        gap = summary.get('total_gap', 0)

        labels = ['Actual Loss', 'Projected Loss', 'Gap']
        values = [actual, projected, gap]
        colors = ['red', 'orange', 'gray']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=colors)

        plt.title("Charged-Off Loan Loss: Actual vs Projected")
        plt.ylabel("Loss Amount ($)")
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Add value labels above bars
        for bar in bars:
            yval = bar.get_height()
            label = f"${yval:,.0f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + yval * 0.02,
                label,
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.tight_layout()
        plt.show()
    def plot_estimated_loss_for_late_loans(self):
        """
        Plots a histogram of estimated losses for currently late loans.
        """
        df = self.eda_data_frame.copy()

        # Filter to late loans only
        late_loans = df[df['loan_status'].str.contains('Late', na=False)].copy()

        # Calculate estimated loss if not already done
        if 'total_paid' not in late_loans.columns:
            late_loans['total_paid'] = (
                late_loans['total_rec_prncp'].fillna(0) +
                late_loans['total_rec_int'].fillna(0) +
                late_loans['total_rec_late_fee'].fillna(0) +
                late_loans['recoveries'].fillna(0)
            )

        late_loans['estimated_remaining'] = late_loans['total_payment'] - late_loans['total_paid']
        late_loans['estimated_loss'] = late_loans['funded_amount'] - late_loans['total_paid'] - late_loans['estimated_remaining']
        late_loans['estimated_loss'] = late_loans['estimated_loss'].clip(lower=0)

        plt.figure(figsize=(10, 6))
        plt.hist(late_loans['estimated_loss'].dropna(), bins=100, color='orange', edgecolor='black', alpha=0.7)
        plt.title('Estimated Loss Distribution for Late Loans')
        plt.xlabel('Estimated Loss Amount')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Format x-axis with dollar values
        formatter = ticker.FuncFormatter(lambda x, _: f'${x:,.0f}')
        plt.gca().xaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.show()
    
    def plot_categorical_comparison(self, categorical_columns, hue_col='risk_group'):
        """
        Plots grouped bar charts (countplot) comparing categorical distributions across groups.

        Parameters:
            categorical_columns (list): List of categorical columns to compare.
            hue_col (str): The column to use for grouping (e.g., 'risk_group').
        """
        for col in categorical_columns:
            if col in self.eda_data_frame.columns:
                plt.figure(figsize=(8, 5))

                # Define fixed order for grade if it's the column being plotted
                order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] if col == 'grade' else None

                sns.countplot(data=self.eda_data_frame, x=col, hue=hue_col, order=order)
                plt.title(f'{col.capitalize()} Distribution by {hue_col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Column '{col}' not found in DataFrame.")


class DataFrameTransform:

    def __init__(self, eda_data_frame):
        """
        Initialize the DataFrameTransform class.

        Args:
            eda_data_frame (pd.DataFrame): The input DataFrame to be transformed.
        """
        self.eda_data_frame = eda_data_frame
        self.no_rows = len(self.eda_data_frame)
        self.transformed_dataframes = {}

        # Box-Cox-specific parameters for inversion
        self.box_cox_lambdas = {}
        self.box_cox_adjustments = {}

        #line to prevent AttributeError
        self.yeo_johnson_transformers = {}

        # Forward transformation mapping
        self.transformation_methods = {
            'log': self.log_transformation,
            'cube_root': self.cube_root_transformation,
            'seventh_root': self.seventh_root_transformation,
            'twenty_first_root': self.twenty_first_root_transformation,
            'untransformed': self.untransformed_data,
            'arcsinh': self.arcsinh_transformation,
            'arcosh': self.arcosh_transformation,
            'box_cox': self.box_cox_transformation,
            'yeo_johnson': self.yeo_johnson_transformation
        }

        # Inverse transformation mapping
        self.inverse_transformation_methods = {
            'log': self.inverse_log_transformation,
            'cube_root': self.inverse_cube_root_transformation,
            'seventh_root': self.inverse_seventh_root_transformation,
            'twenty_first_root': self.inverse_twenty_first_root_transformation,
            'arcsinh': self.inverse_arcsinh_transformation,
            'arcosh': self.inverse_arcosh_transformation,
            'box_cox': self.inverse_box_cox_transformation,
            'yeo_johnson': self.inverse_yeo_johnson_transformation
        }
        
    def drop_column(self, column_name):
        """
        Drops a specific column from the DataFrame if it exists.
        """
        if column_name in self.eda_data_frame.columns:
            self.eda_data_frame.drop(columns=[column_name], inplace=True)
            print(f"Column '{column_name}' has been dropped.")
        else:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
        
        return self.eda_data_frame

    def apply_selected_transforms(self, column_transform_map: dict) -> pd.DataFrame:
        """
        Applies selected transformations to specific columns in the DataFrame.

        Parameters:
            column_transform_map (dict): Dictionary mapping column names to a list of transformation names to apply.
                                        Example: {'loan_amount': ['yeo_johnson'], 'id': ['seventh_root']}

        Returns:
            pd.DataFrame: A new DataFrame with transformed values applied to specified columns.
        """
        result_df = self.eda_data_frame.copy()

        # Identify all unique transformation names (excluding 'untransformed')
        requested_transforms = set(
            t for transform_list in column_transform_map.values() for t in transform_list if t != 'untransformed'
        )

        # Execute each transformation method once
        for transform_name in requested_transforms:
            if transform_name in self.transformation_methods:
                self.transformation_methods[transform_name]()

        # Apply transformed values to the specified columns
        for col, transforms in column_transform_map.items():
            for transform_name in transforms:
                if transform_name == 'untransformed':
                    continue

                transformed_df = self.transformed_dataframes.get(transform_name)
                if transformed_df is not None and col in transformed_df.columns:
                    result_df[col] = transformed_df[col]

        return result_df
    
    def apply_inverse_column_transforms(self, column_transform_map):
        """
        Reverses the transformations based on the column_transform_map used during forward transformation.

        Parameters:
            column_transform_map (dict): A dictionary where keys are column names and values
                                        are lists of transformation names applied to those columns.

        Returns:
            pd.DataFrame: DataFrame with inverse transformations applied to specified columns only.
                        Prevents re-adding columns that were manually dropped.
        """
        inverse_df = self.eda_data_frame.copy()
        current_columns = set(inverse_df.columns)  # Track only current cols

        for col, transforms in column_transform_map.items():
            if not transforms or transforms[0] == 'untransformed':
                continue

            transform_name = transforms[0]

            if transform_name not in self.inverse_transformation_methods:
                print(f"[Inverse] No inverse method for '{transform_name}' — skipping column '{col}'")
                continue

            inverse_transformed_df = self.inverse_transformation_methods[transform_name]()

            # Only apply the inverse if:
            # 1. The column was part of the inverse transform result
            # 2. The column currently exists in the DataFrame (i.e., wasn’t manually dropped)
            if col in inverse_transformed_df.columns and col in current_columns:
                inverse_df[col] = inverse_transformed_df[col]
            else:
                print(f"[Inverse] Column '{col}' not applied. Reason: not in current frame or inverse output.")

        return inverse_df

    def export_transformation_state(self):
        """
        Exports the internal transformation state, including:
        - Transformed data
        - Box-Cox lambdas and adjustments
        - Yeo-Johnson fitted transformer
        - Original column names
        """
        return {
            "transformed_dataframes": self.transformed_dataframes,
            "box_cox_lambdas": self.box_cox_lambdas,
            "box_cox_adjustments": self.box_cox_adjustments,
            "yeo_johnson_transformers": self.yeo_johnson_transformers,
            "original_columns": list(self.eda_data_frame.columns)
        }

    def import_transformation_state(self, state):
        """
        Restores previously saved transformation state.
        
        Parameters:
            state (dict): Dictionary of transformation metadata.
        """
        self.transformed_dataframes = state.get("transformed_dataframes", {})
        self.box_cox_lambdas = state.get("box_cox_lambdas", {})
        self.box_cox_adjustments = state.get("box_cox_adjustments", {})
        self.yeo_johnson_transformers = state.get("yeo_johnson_transformers", {})

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
    
    def remove_strand_outlier(self, column, bin_count=100, frequency_ratio_threshold=5.0):
        """
        Removes rows corresponding to a single-bin spike in the histogram of a column.
        
        Parameters:
            column (str): The column to analyze.
            bin_count (int): Number of histogram bins (default: 100).
            frequency_ratio_threshold (float): How much larger the max bin can be relative to the median to be flagged (default: 5.0).
            
        Returns:
            pd.DataFrame: DataFrame with strand outlier rows removed.
        """
        col_data = self.eda_data_frame[column].dropna()

        # Generate histogram
        frequencies, bin_edges = np.histogram(col_data, bins=bin_count)

        max_freq_index = np.argmax(frequencies)
        max_freq = frequencies[max_freq_index]
        median_freq = np.median(frequencies)

        # Check if it's a spike
        if max_freq > frequency_ratio_threshold * median_freq:
            bin_start = bin_edges[max_freq_index]
            bin_end = bin_edges[max_freq_index + 1]

            mask = ~((self.eda_data_frame[column] >= bin_start) & (self.eda_data_frame[column] < bin_end))
            removed_count = (~mask).sum()
            print(f"[Strand Outlier Removal] Removed {removed_count} rows from '{column}' in bin range [{bin_start:.2f}, {bin_end:.2f}]")

            self.eda_data_frame = self.eda_data_frame[mask]
        else:
            print(f"[Strand Outlier Removal] No dominant spike found in '{column}'.")

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

    def untransformed_data(self):
        """
        Stores a copy of the original untransformed DataFrame in the
        `transformed_dataframes` dictionary under the key 'untransformed'.
        """
        # Ensure that 'transformed_dataframes' is initialized
        if not hasattr(self, 'transformed_dataframes'):
            self.transformed_dataframes = {}

        # Store a deep copy of the original DataFrame
        self.transformed_dataframes['untransformed'] = self.eda_data_frame.copy()

    def twenty_first_root_transformation(self):
        """
        Applies a 21st root transformation to all numerical columns in the DataFrame.
        Stores the result in the `transformed_dataframes` dictionary under the key 'cube_root'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            df_transformed[col] = df_transformed[col].apply(lambda x: x**(1/21) if pd.notnull(x) else x)

        self.transformed_dataframes['twenty_first_root'] = df_transformed

    def seventh_root_transformation(self):
        """
        Applies a seventh root transformation to all numerical columns in the DataFrame.
        Stores the result in the `transformed_dataframes` dictionary under the key 'cube_root'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:

            df_transformed[col] = df_transformed[col].apply(lambda x: x**(1/7) if pd.notnull(x) else x)

        self.transformed_dataframes['seventh_root'] = df_transformed

    def box_cox_transformation(self):
        df_transformed = self.eda_data_frame.copy()
        self.box_cox_lambdas = {}
        self.box_cox_adjustments = {}

        for col in df_transformed.select_dtypes(include='number').columns:
            col_data = df_transformed[col].dropna()

            if col_data.nunique() <= 1:
                print(f"[Box-Cox] Skipping column '{col}' (constant after cleaning)")
                continue

            min_value = col_data.min()
            adjustment = 0

            if min_value <= 0:
                adjustment = abs(min_value) + 1
                df_transformed[col] = df_transformed[col] + adjustment

            df_transformed[col] = df_transformed[col].astype(float)  # Ensure dtype compatibility

            try:
                transformed, fitted_lambda = boxcox(df_transformed[col].dropna())
                # Safely assign transformed values
                df_transformed.loc[df_transformed[col].dropna().index, col] = transformed
                self.box_cox_lambdas[col] = fitted_lambda
                self.box_cox_adjustments[col] = adjustment
            except ValueError as e:
                print(f"[Box-Cox] Skipping column '{col}' due to error: {e}")
                continue

        self.transformed_dataframes['box_cox'] = df_transformed

    def yeo_johnson_transformation(self):
        """
        Applies a Yeo-Johnson transformation to all numerical columns in the DataFrame.
        The Yeo-Johnson transformation can handle both positive and negative values.
        Stores the result in the `transformed_dataframes` dictionary under the key 'yeo_johnson'.
        Also stores individual transformers for inverse transformation.
        """
        df_transformed = self.eda_data_frame.copy()

        # Initialize if not already
        if not hasattr(self, 'yeo_johnson_transformers'):
            self.yeo_johnson_transformers = {}

        numeric_cols = df_transformed.select_dtypes(include='number').columns

        for col in numeric_cols:
            reshaped_col = df_transformed[[col]].values  # Keeps 2D shape
            transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            df_transformed[col] = transformer.fit_transform(reshaped_col)
            self.yeo_johnson_transformers[col] = transformer  # Store transformer for inverse

        self.transformed_dataframes['yeo_johnson'] = df_transformed

    def arcosh_transformation(self):
        """
        Applies the inverse hyperbolic cosine (arcosh) transformation to all numerical columns.
        Only valid for values >= 1. Others will result in NaN.
        Stores the result in the `transformed_dataframes` dictionary under the key 'arcosh'.
        """
        df_transformed = self.eda_data_frame.copy()

        for col in df_transformed.select_dtypes(include='number').columns:
            df_transformed[col] = df_transformed[col].apply(lambda x: np.arccosh((x**(0.5))+1))

        self.transformed_dataframes['arcosh'] = df_transformed

    def inverse_log_transformation(self):
        """
        Inverts the log transformation for all numerical columns.
        Adjusts using the same offset used during log transform if original data contained non-positive values.
        
        Returns:
            pd.DataFrame: Inverse log-transformed DataFrame.
        """
        if 'log' not in self.transformed_dataframes:
            raise ValueError("Log transformation not found.")

        df = self.transformed_dataframes['log'].copy()

        for col in df.select_dtypes(include='number').columns:
            original_min = self.eda_data_frame[col].min()
            adjustment = abs(original_min) + 1 if original_min <= 0 else 0
            df[col] = df[col].apply(lambda x: np.exp(x) - adjustment if pd.notnull(x) else x)

        return df


    def inverse_cube_root_transformation(self):
        """
        Inverts the cube root transformation for all numerical columns.

        Returns:
            pd.DataFrame: Inverse cube-root-transformed DataFrame.
        """
        if 'cube_root' not in self.transformed_dataframes:
            raise ValueError("Cube root transformation not found.")

        df = self.transformed_dataframes['cube_root'].copy()

        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].apply(lambda x: x ** 3 if pd.notnull(x) else x)

        return df


    def inverse_seventh_root_transformation(self):
        """
        Inverts the seventh root transformation for all numerical columns.

        Returns:
            pd.DataFrame: Inverse seventh-root-transformed DataFrame.
        """
        if 'seventh_root' not in self.transformed_dataframes:
            raise ValueError("Seventh root transformation not found.")

        df = self.transformed_dataframes['seventh_root'].copy()

        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].apply(lambda x: x ** 7 if pd.notnull(x) else x)

        return df


    def inverse_twenty_first_root_transformation(self):
        """
        Inverts the twenty-first root transformation for all numerical columns.

        Returns:
            pd.DataFrame: Inverse 21st-root-transformed DataFrame.
        """
        if 'twenty_first_root' not in self.transformed_dataframes:
            raise ValueError("Twenty-first root transformation not found.")

        df = self.transformed_dataframes['twenty_first_root'].copy()

        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].apply(lambda x: x ** 21 if pd.notnull(x) else x)

        return df


    def inverse_arcsinh_transformation(self):
        """
        Inverts the arcsinh transformation for all numerical columns using the sinh function.

        Returns:
            pd.DataFrame: Inverse arcsinh-transformed DataFrame.
        """
        if 'arcsinh' not in self.transformed_dataframes:
            raise ValueError("Arcsinh transformation not found.")

        df = self.transformed_dataframes['arcsinh'].copy()

        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].apply(lambda x: np.sinh(x) if pd.notnull(x) else x)

        return df


    def inverse_arcosh_transformation(self):
        """
        Inverts the arcosh transformation for all numerical columns.
        Uses the identity: x = (cosh(y) - 1)^2

        Returns:
            pd.DataFrame: Inverse arcosh-transformed DataFrame.
        """
        if 'arcosh' not in self.transformed_dataframes:
            raise ValueError("Arcosh transformation not found.")

        df = self.transformed_dataframes['arcosh'].copy()

        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].apply(lambda x: (np.cosh(x) - 1) ** 2 if pd.notnull(x) else x)

        return df


    def inverse_box_cox_transformation(self):
        """
        Inverts the Box-Cox transformation using the stored lambda values and pre-shift adjustments.
        Uses scipy.special.inv_boxcox.

        Returns:
            pd.DataFrame: Inverse Box-Cox-transformed DataFrame.
        """
        if 'box_cox' not in self.transformed_dataframes:
            raise ValueError("Box-Cox transformation not found.")

        if not hasattr(self, 'box_cox_lambdas'):
            raise ValueError("Box-Cox lambdas not stored. Cannot invert.")

        df = self.transformed_dataframes['box_cox'].copy()

        for col in df.select_dtypes(include='number').columns:
            if col not in self.box_cox_lambdas:
                print(f"[Inverse Box-Cox] Lambda not found for '{col}', skipping.")
                continue

            lam = self.box_cox_lambdas[col]
            adjustment = self.box_cox_adjustments.get(col, 0)

            try:
                df[col] = inv_boxcox(df[col], lam) - adjustment
            except Exception as e:
                print(f"[Inverse Box-Cox] Failed for column '{col}': {e}")

        return df


    def inverse_yeo_johnson_transformation(self):
        """
        Inverts the Yeo-Johnson transformation using stored transformers per column.

        Returns:
            pd.DataFrame: Inverse-transformed DataFrame.
        """
        if 'yeo_johnson' not in self.transformed_dataframes:
            raise ValueError("Yeo-Johnson transformation not found.")

        df = self.transformed_dataframes['yeo_johnson'].copy()
        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            if col not in self.yeo_johnson_transformers:
                print(f"[Inverse Yeo-Johnson] Missing transformer for column: {col}")
                continue
            try:
                reshaped = df[[col]].values
                df[col] = self.yeo_johnson_transformers[col].inverse_transform(reshaped)
            except Exception as e:
                print(f"[Inverse Yeo-Johnson] Inversion failed for column '{col}': {e}")

        return df


    def apply_all_transforms(self):
        """
        Applies all 9 transformations to the DataFrame.

        The transformations include:
            - Untransformed Data
            - Log Transformation
            - Cube Root Transformation
            - Seventh Power Root
            - 21 st Power Root
            - Arcsinh Transformation
            - Arcosh Transformation
            - Box-Cox Transformation
            - Yeo-Johnson Transformation
        

        Returns:
            list: A list of transformed DataFrames.
        """
        self.untransformed_data()

        self.log_transformation()

        self.cube_root_transformation()

        self.seventh_root_transformation()

        self.twenty_first_root_transformation()

        self.arcsinh_transformation()

        self.arcosh_transformation()

        self.box_cox_transformation()

        self.yeo_johnson_transformation()

        return [self.transformed_dataframes[key] for key in self.transformed_dataframes.keys()]
