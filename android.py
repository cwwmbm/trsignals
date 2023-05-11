import pandas as pd
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from s_ch import signal_check

class MyBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MyBoxLayout, self).__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Create a Button
        self.button = Button(text='Execute Function')
        self.button.bind(on_press=self.execute_function)
        self.add_widget(self.button)
        
        # Create a placeholder for the ScrollView
        self.scroll_view = None
    
    def execute_function(self, instance):        
        # Call the function and display the result in the GridLayout
        df = signal_check()
        if self.scroll_view:
            self.remove_widget(self.scroll_view)
        
        # Define a dictionary that maps column names to their corresponding widths
        column_widths = {
            'Symbol': 200,
            'Name': 200,
            'Description': 1000,
            'Days': 200,
            'Profit': 200,
            'Buy?': 200,
            'Sell?': 200,
            'Hold?': 200,
            'Date': 200,
            'Verdict': 1000
        }
        
        # Create a ScrollView to allow scrolling if the table is large
        self.scroll_view = ScrollView(
            size_hint=(1, None),
            size=(self.width, self.height),
            bar_width=10,  # Width of the scrollbar
            scroll_type=['bars', 'content'],  # Enable both horizontal and vertical scrolling
            bar_color=[0.7, 0.7, 0.7, 1],  # Color of the scrollbar
            bar_inactive_color=[0.9, 0.9, 0.9, 1]  # Color of the scrollbar when not scrolling
        )
        
        # Create the GridLayout
        grid_layout = GridLayout(
            cols=len(df.columns),
            size_hint_y=None,
            size_hint_x=None,  # Allow the GridLayout to have a fixed width
            width=sum(column_widths.values()),  # Set the width based on the sum of column widths
            spacing=5
        )
        grid_layout.bind(minimum_height=grid_layout.setter('height'))
        
        # Function to update the height of the label based on the texture size
        def update_label_height(label, texture_size):
            label.height = texture_size[1]
        
        # Add column headers to the GridLayout
        for col_name in df.columns:
            label = Label(
                text=col_name,
                bold=True,
                size_hint_y=None,
                text_size=(column_widths[col_name], None),
                halign='left'
            )
            label.bind(texture_size=update_label_height)
            grid_layout.add_widget(label)
        
        # Add data to the GridLayout
        for _, row in df.iterrows():
            for col_name, value in row.items():
                label = Label(
                    text=str(value),
                    size_hint_y=None,
                    text_size=(column_widths[col_name], None),
                    halign='left'
                )
                label.bind(texture_size=update_label_height)
                grid_layout.add_widget(label)
