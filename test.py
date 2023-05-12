import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

data = pd.read_csv('data.csv')
print (data.head(5))