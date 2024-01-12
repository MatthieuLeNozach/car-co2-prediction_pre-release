import pandas as pd
import io
import re


########## Project color style ##########
def generate_styles(header_color='steelblue', odd_row_color='aliceblue', even_row_color='white'):
    styles = [
        {'selector': 'th', 'props': [('background-color', header_color), ('color', 'white')]},  # HEADER
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', odd_row_color)]},  # ODD ROWS
        {'selector': 'tr:nth-of-type(even)', 'props': [('background-color', even_row_color)]}  # EVEN ROWS
    ]
    return styles


########## End project style



########## Table Tools ##########
def displayer(df, n=5, styles=None): # PANDAS (style)
    if styles is None:
        styles = generate_styles()
    
    format_dict = {col: "{:.3f}" for col in df.select_dtypes('float').columns}
    
    styled_df = df.head(n).style.format(format_dict).set_table_styles(styles)
    display(styled_df)
    

def display_info(df, styles=None): # PANDAS (style)
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
    
    displayer(column_df, n=len(column_df), styles=styles)
    displayer(general_df, n=len(general_df), styles=styles)
    
    
def display_describe(df, styles=None): # PANDAS (style)
    describe_df = df.describe().transpose().reset_index()
    describe_df.columns = ['Colonne', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    displayer(describe_df, n=len(describe_df), styles=styles)





"""
def display_na(df): # PLOTLY METHOD
    proportions = df.isna().mean().sort_values(ascending=False)*100
    proportions = proportions.round(2)

    # Create a table
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Colonne', 'Valeurs manquantes (%)'],
                    fill_color='SteelBlue',
                    align='left',
                    font=dict(color='white', size=20)),
        cells=dict(values=[proportions.index, proportions.values],
                fill_color='PowderBlue',
                align='left',
                font=dict(size=12))
    )])

    fig.update_layout(height=1100)
    fig.show()
"""

def display_na(df): # PANSAS (style)
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
    
    # Format the percentages
    styled_df.format({'Valeurs manquantes (%)': '{:,.2f}%'})
    
    return styled_df


########## End of table tools ##########