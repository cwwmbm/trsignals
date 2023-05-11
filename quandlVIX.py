import quandl
import pandas as pd
import getdata as gt


def get_quandl_data():
    # Set your API key
    quandl.ApiConfig.api_key = '3hAmwu7_u6y5g4P37Sbo'
    # Get VIX futures data
    #vx1 = quandl.get('CHRIS/CBOE_VX1', start_date='2000-01-01', end_date='2023-04-11')
    #vx4 = quandl.get('CHRIS/CBOE_VX4', start_date='2000-01-01', end_date='2023-04-11')
    data = quandl.get('CHRIS/CME_NQ1', start_date='2000-01-01', end_date='2023-04-20')

    data = gt.normalize_dataframe(data)
    data = data[['Date', 'Open', 'High', 'Low', 'Last', 'Volume']]
    data.rename(columns={'Last': 'Close'}, inplace=True)
    #print(data)
    return data

# Calculate the Contango1-2
#vx1['Contango1-2'] = ((vx1['Settle'] - vx4['Settle']) / vx1['Settle']) * 100
""""
# Merge both dataframes
vix_term_structure = pd.concat([vx1['Contango1-2'], vx4['Settle']], axis=1)
vix_term_structure.columns = ['Contango1-2', 'VX4']
vix_term_structure.dropna(inplace=True)
# Display the data
print(vix_term_structure)
"""