from IPython.display import display, HTML
import numpy as np
import pandas as pd
import os
import datetime
#from datetime import datetime
import io
import re
import plotly.graph_objects as go
from sklearn.metrics import (classification_report, mean_squared_error,
                             mean_absolute_error, r2_score)



########## Project color style ##########
def generate_styles(header_color='steelblue', odd_row_color='aliceblue', even_row_color='white'):
    """Generate Pandas Styler styles for HTML display.
    
    Sets background colors for header, odd rows, and even rows.
    
    Args:
      header_color: Background color for header row.
      odd_row_color: Background color for odd numbered rows.
      even_row_color: Background color for even numbered rows.
      
    Returns:
      styles: List of style dictionaries for Pandas Styler.
    """
    styles = [
        {'selector': 'th', 'props': [('background-color', header_color), ('color', 'white')]},  # HEADER
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', odd_row_color)]},  # ODD ROWS
        {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', even_row_color)]}  # EVEN ROWS
    ]
    return styles


########## End project style



########## Table Tools ##########
def displayer(df, n=5, styles=None, title=None, save=True):  # Added title parameter
    """Display a Pandas DataFrame or Styler as an HTML table.
    
    Apply formatting and styling to the DataFrame before 
    displaying as HTML table. Styling includes background 
    colors for header, odd, and even rows. 
    
    Args:
      df: Pandas DataFrame to display.
      n: Number of rows to display.
      styles: List of style dicts for Styler.
      title: Optional title displayed above table.
      save: Whether to save the styled DataFrame to file.
    
    Returns:
      None. Displays styled DataFrame as HTML table.
    """
    print("Save:", save)
    if styles is None:
        styles = generate_styles()
    
    if isinstance(df, pd.DataFrame):
        format_dict = {col: "{:.3f}" for col in df.select_dtypes('float').columns}
        styled_df = df.head(n).style.format(format_dict).set_table_styles(styles)
    else:  # Assume it's a Styler object
        styled_df = df
    
    if title is not None:  # If a title is provided, set it
        styled_df = styled_df.set_caption(f'<h2>{title}</h2>')  # Use HTML tags to increase the size
    
    if save:
        save_styled_df(styled_df, title)
    
    display(styled_df)
    print(type(styled_df))
    

def display_info(df, styles=None, title=None, save=True): # PANDAS (style)
    """
    Display Pandas DataFrame information.
    
    Extracts column and general information from df.info() output,
    formats into DataFrames and displays as styled HTML tables.
    
    Args:
      df: Pandas DataFrame.
      styles: List of style dicts for Styler.
      title: Optional title displayed above tables. 
      save: Whether to save tables to file.
    
    Returns:
      None. Displays column info and general info as HTML tables.
    """
    if styles is None:
        styles = generate_styles()      
    
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    lines = info_str.split("\n")
    
    # Infos des colonnes
    column_info = lines[5:-3]
    column_df = pd.DataFrame([re.split(r'\s\s+', line.strip()) for line in column_info])
    column_df = column_df.iloc[:, 1:]
    column_names = ['Column', 'Non-Null Count', 'Dtype']
    column_df.columns = column_names[:len(column_df.columns)]
    
    # Infos générales
    general_info = lines[1:3] + lines[-4:-1]
    general_df = pd.DataFrame(general_info, columns=['Info'])
    
    styled_column_df = column_df.style.set_table_styles(styles)
    styled_general_df = general_df.style.set_table_styles(styles)
    
    displayer(column_df, n=len(column_df), styles=styles, title=title, save=save)
    displayer(general_df, n=len(general_df), styles=styles, title=title, save=save)
    
    
def display_describe(df, styles=None, title=None, save=True): # PANDAS (style)
    """Display descriptive statistics for a Pandas DataFrame.
    
    Generates a descriptive statistics summary using df.describe() and 
    displays it as a nicely formatted table. Useful for quickly 
    examining the distribution of data in a DataFrame.
    """
    describe_df = df.describe().transpose().reset_index()
    describe_df.columns = ['Colonne', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    displayer(describe_df, n=80, styles=styles, save=save, title=title)



def display_na(df, title=None, save=True): # PANSAS (style)
    """Displays missing values in a Pandas DataFrame as a formatted HTML table.
    
    Calculates the percentage of missing values for each column, sorts by percentage, 
    and displays as a formatted Pandas Styler table with bar charts indicating percentage.
    Table style includes colored header and alternating row colors.
    
    Args:
      df: Pandas DataFrame to analyze for missing values.
      title: Optional string table title.
      save: Whether to save table HTML to file.
    
    Returns:
      None. Displays and optionally saves HTML table.
    """
    proportions = df.isna().mean().sort_values(ascending=False)*100
    proportions = proportions.round(2).to_frame().reset_index()
    proportions.columns = ['Colonne', 'Valeurs manquantes (%)']
    
    # Create a styled DataFrame
    styled_df = proportions.style.bar(subset=['Valeurs manquantes (%)'], align='left', color=['lightcoral', 'lightcoral'])
    
    # Set table styles
    styles = [
        {'selector': 'th', 'props': [('background-color', 'steelblue'), ('color', 'white')]},  # Header color
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', 'aliceblue')]},  # Row colors
        {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', 'white')]}
    ]
    styled_df = styled_df.set_table_styles(styles)
    styled_df.format({'Valeurs manquantes (%)': '{:,.2f}%'})
    
    displayer(styled_df, n=len(df), styles=styles, title=title, save=save)

    
def save_styled_df(styled_df, name=None):
    """Saves a Pandas Styler DataFrame to an HTML file.
    
    Args:
      styled_df: Styled Pandas DataFrame to save.
      name: Name to use for saved file. If None, uses 'untitled'.
    
    Returns:
      None. Saves HTML file to ../output/tables.
    """
    os.makedirs("../output/tables", exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = name if name is not None else 'untitled'
    base_filename = re.sub('[^a-zA-Z0-9-_]', '_', base_filename)
    filename = f"{base_filename}_{timestamp}"
    
    html = styled_df.to_html()
    with open(f"../output/tables/{filename}.html", "w", encoding='utf-8') as f:  # 'utf-8' pour afficher les accents
        f.write(html)
    print(f"Saved {name} styled DataFrame to output/tables")
########## End of table tools ##########


########## ML Tables ##########

def display_classification_report(y_true, y_pred, title=None, styles=None): 
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    if styles is None:
        styles = generate_styles()
    
    displayer(report_df, n=len(report_df), styles=styles, title=title)
    
    
def display_feature_importances(model, data, title=None):
    feats = {}
    for feature, importance in zip(data.columns, model.feature_importances_):
        feats[feature] = importance

    importances  = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0:'importance'})
    importances = importances.sort_values(by='importance', ascending=False)
    displayer(importances, n=None, title=title)
    
    
def display_regression_report(actual_values, predicted_values, set_name):
    """
    Calculates and returns regression metrics for a set of actual and predicted values.
    
    Parameters:
    - actual_values: Array-like of actual target values.  
    - predicted_values: Array-like of predicted target values.
    - set_name: Name of the data set.
    
    Returns:
    Pandas DataFrame containing:
    - Mean Squared Error
    - Root Mean Squared Error  
    - Mean Absolute Error
    - R2 Score
    """
    mean_squared_err = mean_squared_error(actual_values, predicted_values)
    root_mean_squared_err = np.sqrt(mean_squared_err)  # Corrected
    mean_absolute_err = mean_absolute_error(actual_values, predicted_values)
    r2_scoring = r2_score(actual_values, predicted_values)
    
    # Create a DataFrame with the calculated metrics
    report = pd.DataFrame({
        set_name: [mean_squared_err, root_mean_squared_err, mean_absolute_err, r2_scoring]
    }, index=['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R2 Score'])
    
    return report

def display_combined_report(y_train, y_pred_train, y_test, y_pred_test, title=None, styles=None):
    """Displays a combined regression report for train and test sets.
    
    Calculates regression metrics for train and test sets using 
    display_regression_report(). Concatenates the resulting DataFrames 
    and displays the combined report.
    
    Parameters:
    y_train: Array of actual train target values.  
    y_pred_train: Array of predicted train target values.
    y_test: Array of actual test target values.
    y_pred_test: Array of predicted test target values.  
    title: Optional string title for the combined report.
    styles: Optional custom styles for the report display.
    """
    # Generate reports for train and test sets
    train_report = display_regression_report(y_train, y_pred_train, 'Train Set')
    test_report = display_regression_report(y_test, y_pred_test, 'Test Set')

    # Combine the reports
    combined_report = pd.concat([train_report, test_report], axis=1)

    # Display the combined report
    displayer(combined_report, n=len(combined_report), styles=styles, title=title)
    
    
    