import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Function to load karate movement data from MongoDB
def load_data():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')  # Change to your MongoDB connection URL
    db = client['karate']  # Change 'karate' to your database name
    collection = db['classifications']  # Change 'classifications' to your collection name
    
    data = list(collection.find())
    
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    return df

# Load data
df = load_data()

# Convert lists to tuples if any exist in the 'class' column
if df['class'].apply(lambda x: isinstance(x, list)).any():
    df['class'] = df['class'].apply(lambda x: tuple(x) if isinstance(x, list) else x)

# Sidebar selection
option = st.sidebar.selectbox(
    'Silakan pilih:',
    ('Gerakan', 'Jenis Kelamin', 'Daerah', 'Date')
)

if option == 'Gerakan':
    st.write("""## Gerakan""")
    
    # Map class names
    class_mapping = {
        'Dachi': 'Kuda-Kuda(Dachi)', 
        'Zuki': 'Pukulan(Zuki)', 
        'Uke': 'Tangkisan(Uke)', 
        'Geri': 'Tendangan(Geri)',
        'TidakDiketahui': 'Tidak Diketahui'
    }
    df['class_mapped'] = df['class'].map(class_mapping).fillna(df['class'])

    # Filter the data to only include the specified classes
    allowed_classes = ['Tendangan(Geri)', 'Kuda-Kuda(Dachi)', 'Tidak Diketahui', 'Tangkisan(Uke)']
    filtered_df = df[df['class_mapped'].isin(allowed_classes)]

    # Count movements
    total_gerakan = filtered_df['class_mapped'].value_counts().sum()
    
    # Display total movements
    st.write(f"Total Gerakan: {total_gerakan}")

    # Prepare data for chart
    chart_data = filtered_df['class_mapped'].value_counts().reset_index()
    chart_data.columns = ['class', 'count']
    
    # Display bar chart
    st.bar_chart(chart_data.set_index('class'))
    
    # Display movements table
    st.table(chart_data)

    # Count total movements for all classes
    total_per_class = chart_data['count'].sum()
    
    # Display total movements for all classes
    st.write(f"Total Jumlah dari Semua Gerakan: {total_per_class}")

elif option == 'Jenis Kelamin':
    st.write("""## Jenis Kelamin""")
    
    # Prepare data for gender
    jenis_kelamin_data = df['jenis_kelamin'].value_counts().reset_index()
    jenis_kelamin_data.columns = ['jenis_kelamin', 'count']
    
    # Display bar chart
    st.bar_chart(jenis_kelamin_data.set_index('jenis_kelamin'))
    
    # Display gender table
    st.table(jenis_kelamin_data)

elif option == 'Daerah':
    st.write("""## Daerah""")
    
    # Map region names
    daerah_mapping = {
        'tegal timur': 'Tegal Timur', 
        'tegal barat': 'Tegal Barat', 
        'tegal selatan': 'Tegal Selatan', 
        'margadana': 'Margadana'
    }
    df['daerah_mapped'] = df['daerah'].map(daerah_mapping).fillna(df['daerah'])
    
    # Prepare data for region
    daerah_data = df['daerah_mapped'].value_counts().reset_index()
    daerah_data.columns = ['daerah', 'count']
    
    # Display bar chart
    st.bar_chart(daerah_data.set_index('daerah'))
    
    # Display region table
    st.table(daerah_data)

elif option == 'Date':
    st.write("""## Date""")
    
    # Get current date
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    
    # Display current date
    st.write(f"Data berdasarkan Date: {current_date}")

    # Example line chart with sample data
    data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'value': [3, 6, 2, 7, 5, 8, 3, 6, 2, 7]
    })
    
    st.line_chart(data.set_index('date'))