import pandas as pd
import plotly.express as px
import streamlit as st
import subprocess
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.decomposition import PCA
from streamlit_metrics import metric, metric_row

# st.set_page_config(
#     page_title="Nomin Supplier Dynamic Scoring",
#     page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSP-FdJah9dGxPHqBmdDi-hkkqO_QmlpVIWHg&s",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
# #aaa
# pd.set_option('display.float_format', lambda x: '%.3f' % x)

# tab1, tab2, tab3 = st.tabs(["Үндсэн үзүүлэлтүүд", "Категорийн оноо", "Кластерын оноо"])


st.image("https://erp15.nomin.mn/web/image/125495-0f10b8a0/%D0%9D%D0%BE%D0%BC%D0%B8%D0%BD%20%D0%9B%D0%BE%D0%B3%D0%BE.png", width=150)  # Adjust width as needed
st.markdown("<br>", unsafe_allow_html=True)

odf = pd.read_csv('data/orderData_category.csv')
pdf = pd.read_csv('data/purchaseData_category.csv')

odf.columns = ['year',
'day',
'customer',
'custid',
'category',
'divCnt',
'skuCnt',
'orderQty',
'avgPrice',
'avgCommiss',
'totalAmt',
'orderCnt',
'manualOrder',
'autoOrder',
'avgLeadTime',
'confirmed']

pdf.columns = ['year','day','customer','custid','category','divcnt','skucnt','inqty','outqty','avgprice','avgcomiss',
              'avgtaxedprice','inamt','outamt','intaxedamt','outtaxedamt','orderqty','purchasecnt','correct','missing','surplus','purchaseonly']
pdf['customer'] = pdf['customer'].astype('str')
odf['customer'] = odf['customer'].astype('str')
pdf = pdf[~pdf['customer'].str.contains('Номин')]
odf = odf[~odf['customer'].str.contains('Номин')]
pdf = pdf[~pdf['customer'].str.contains('НОМИН')]
odf = odf[~odf['customer'].str.contains('НОМИН')]
odf['date'] = pd.to_datetime(odf['year'].astype(str) + odf['day'].astype(str).str.zfill(3), format='%Y%j')
pdf['date'] = pd.to_datetime(pdf['year'].astype(str) + pdf['day'].astype(str).str.zfill(3), format='%Y%j')
odf = odf.dropna()
pdf = pdf.dropna()
pdf['errors_qty'] = pdf['missing'] + pdf['surplus'] 
pdf['errors_cnt'] = (pdf['errors_qty'] > 0).astype(int)
pdf['profit'] = pdf['avgcomiss']/100
pdf['profit'] = pdf['intaxedamt'] * pdf['profit']
# Aggregate data by customer
order_data = odf.groupby(['customer','category']).agg({
    'divCnt': 'max',
    'skuCnt': 'max',
    'orderQty': 'sum',
    'avgPrice': 'median',
    'totalAmt': 'sum',
    'orderCnt': 'sum',
    'manualOrder': 'sum',
    'autoOrder': 'sum',
    'avgLeadTime': 'median',
    'confirmed': 'sum'
}).reset_index()

purchase_data = pdf.groupby(['customer','category']).agg({
    'divcnt': 'max',
    'skucnt': 'max',
    'purchasecnt':'sum',
    'inqty':'sum',
    'outqty':'sum',
    'orderqty':'sum',
    'profit':'sum',
    'avgcomiss':'median',
    'avgprice': 'median',
    'inamt': 'sum',
    'intaxedamt': 'sum',
    'errors_qty': 'sum',
    'errors_cnt': 'sum',
}).reset_index()
df = purchase_data.merge(order_data,on=['customer','category'],how='left')
df = df.drop(columns = ['skuCnt','divCnt','outqty','avgPrice','orderQty','avgPrice','totalAmt','orderCnt','confirmed'])
df['avg_error'] = df['errors_cnt'] / df['purchasecnt']
df = df.fillna(value=0)
df['profit'] = df['profit'].astype('int64')
category_counts = df.groupby('category')['customer'].nunique()
single_customer_categories = category_counts[category_counts == 1].index.tolist()
df = df[~df['category'].isin(single_customer_categories)]
with st.expander("Тусгаарласан барааны ангилалууд"):
    # Create a scrollable div
    scrollable_div = """
    <div style="max-height: 200px; overflow-y: auto;">
        <ul>
    """

    for i in single_customer_categories:
        scrollable_div += f"<li>{i}</li>"

    scrollable_div += """
        </ul>
    </div>
    """

    st.markdown(scrollable_div, unsafe_allow_html=True)

df = df[['customer', 'category', 'divcnt', 'skucnt', 'purchasecnt', 'inqty',
       'orderqty', 'profit', 'avgcomiss', 'avgprice', 'inamt', 'intaxedamt',
       'errors_qty', 'errors_cnt', 'manualOrder', 'autoOrder', 'avgLeadTime',
       'avg_error']]

metrics = ['profit', 'purchasecnt', 'orderqty', 'avgLeadTime', 'avg_error', 'divcnt', 'skucnt', 'avgcomiss']

def remove_outliers_iqr_within_category(df, columns):
    """Remove outliers using IQR method for specified columns within each category."""
    filtered_out_customers = []  # List to hold filtered-out customers
    def filter_outliers(group):
        # Store the original customers in the group
        original_customers = group['customer'].tolist()

        for column in columns:
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            group = group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]

        # Identify filtered-out customers
        removed_customers = set(original_customers) - set(group['customer'].tolist())
        filtered_out_customers.extend(removed_customers)

        return group
    filtered_df = df.groupby('category').apply(filter_outliers).reset_index(drop=True)
    return filtered_df, filtered_out_customers

if 'weights' not in st.session_state:
    st.session_state.weights = {
        'profit': 0.4,
        'purchasecnt': 0.25,
        'orderqty': 0.1,
        'avgLeadTime': 0.05,
        'avg_error': -0.1,
        'divcnt': 0.025,
        'skucnt': 0.025,
        'avgcomiss': 0.05
    }
weights = st.session_state.weights
display_names = {
    'profit': "Ашгийн хувь",
    'purchasecnt': "ХА тоо",
    'orderqty': "Захиалгын тоо",
    'avgLeadTime': "Дундаж хүргэлтийн хугацаа (хоног)",
    'avg_error': "Алдаатай захиалгын хувь",
    'divcnt': "Нийлүүлдэг салбаруудын тоо",
    'skucnt': "Нийлүүлдэг SKU тоо",
    'avgcomiss': "Дундаж GP"
}

def scale_within_category(group):
    if len(group) > 1:
        scaled_values = scaler.fit_transform(group[metrics])
        return pd.DataFrame(scaled_values, columns=[f'minmax_{col}' for col in metrics], index=group.index)
    else:
        return pd.DataFrame(0, index=group.index, columns=[f'minmax_{col}' for col in metrics])

st.title("Ханган нийлүүлэгчийн барааны төрлийн динамик үнэлгээ")
st.sidebar.header("Жин тохируулах")
input_values = {}

# Create input fields for each weight
for key in weights.keys():
    input_values[key] = st.sidebar.number_input(
        f"{display_names[key]}(%)",
        min_value=-100.0,
        max_value=100.0,
        value=weights[key] * 100,  # Start with the initial weight
        step=0.1
    ,key='category_input') / 100

# Calculate total weight
total_weight = sum(input_values.values())
st.sidebar.write(f"Нийт дүн: {total_weight * 100:.2f}%")

# Check if total weight exceeds 100%
if total_weight > 1.0:
    st.sidebar.warning("Нийт жингийн дүн 100%-аас хэтэрсэн тул засна уу?")
elif total_weight < 1.0:
    st.sidebar.warning("Нийт жингийн дүн 100%-аас бага байж болохгүй тул засна уу?")
else:
    st.sidebar.success("Тохирсон.")

# Initialize session state for weights and selected metrics
if 'weights' not in st.session_state:
    st.session_state.weights = {}
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = []

# Sample input values for weights (you might have a different source)
input_values = {
    'profit': 0.4,
    'purchasecnt': 0.25,
    'orderqty': 0.1,
    'avgLeadTime': 0.05,
    'avg_error': -0.1,
    'divcnt': 0.025,
    'skucnt': 0.025,
    'avgcomiss': 0.05
}
metrics_mapping = {
    "Ашгийн хувь": "profit",
    "ХА тоо": "purchasecnt",
    "Захиалгын тоо": "orderqty",
    "Дундаж хүргэлтийн хугацаа (хоног)": "avgLeadTime",
    "Алдаатай захиалгын хувь": "avg_error",
    "Нийлүүлдэг салбаруудын тоо": "divcnt",
    "Нийлүүлдэг SKU тоо": "skucnt",
    "Дундаж GP": "avgcomiss",
}

# Save weights button
if st.sidebar.button("Хадгалах"):
    for key in weights.keys():
        weights[key] = input_values[key]
    st.session_state.weights = weights
    st.sidebar.success("Амжилттай хадгалагдлаа!")
    st.sidebar.write(weights)

# IQR дундаж тооцоолох баганууд сонгох
st.markdown("IQR дундаж тооцоолох баганууд сонгох")
metrics_mapping = {
    "Ашгийн хувь": "profit",
    "ХА тоо": "purchasecnt",
    "Захиалгын тоо": "orderqty",
    "Дундаж хүргэлтийн хугацаа (хоног)": "avgLeadTime",
    "Алдаатай захиалгын хувь": "avg_error",
    "Нийлүүлдэг салбаруудын тоо": "divcnt",
    "Нийлүүлдэг SKU тоо": "skucnt",
    "Дундаж GP": "avgcomiss",
}

# Define options for the multiselect
options = list(metrics_mapping.keys())

# Get valid default values from session state
valid_defaults = [item for item in st.session_state.selected_metrics if item in options]

# Multiselect for metrics
ms_arr = st.multiselect(
    "2 хэмжүүр сонгох",
    options,
    max_selections=2,
    default=valid_defaults  # Ensure defaults are valid options
)

selected_metrics = [metrics_mapping[item] for item in ms_arr if item in metrics_mapping]
st.session_state.selected_metrics = selected_metrics  # Update session state
st.success("Selected metrics saved: " + ", ".join(selected_metrics))
filtered_df, filtered_out_customers = remove_outliers_iqr_within_category(df, selected_metrics)

scaler = MinMaxScaler()
scaled_df = filtered_df.groupby('category').apply(scale_within_category).reset_index(drop=True)
filtered_df = pd.concat([filtered_df.reset_index(drop=True), scaled_df], axis=1)
filtered_df['score'] = sum(filtered_df[f'minmax_{metric}'] * weights[metric] for metric in metrics)
filtered_df['metric_array'] = filtered_df.apply(
    lambda row: [
        (row[f'minmax_{metric}'] * weight) / row['score'] * 100 if row['score'] != 0 else 0
        for metric, weight in weights.items()
    ], axis=1
)

# Step 1: Calculate average score for each category
avg_scores = filtered_df.groupby('category')['score'].mean().reset_index()
avg_scores.rename(columns={'score': 'cat_avg_score'}, inplace=True)
metric_array_df = pd.DataFrame(filtered_df['metric_array'].tolist())
metric_array_df['category'] = filtered_df['category'].values
avg_metric_array = metric_array_df.groupby('category').mean().reset_index()
# Convert the averages back into a list format
avg_metric_array['avg_metric_array'] = avg_metric_array.iloc[:, 1:].values.tolist()
filtered_df = filtered_df.merge(avg_scores, on='category', how='left')
filtered_df = filtered_df.merge(avg_metric_array[['category', 'avg_metric_array']], on='category', how='left')

# Select customer
selected_customer = st.selectbox("customer", filtered_df.customer.unique(), index=0 if 'selected_customer' not in st.session_state else filtered_df.customer.unique().tolist().index(st.session_state.selected_customer))

if selected_customer:
    st.session_state.selected_customer = selected_customer  # Store selected customer in session state
    fil_df = filtered_df[filtered_df.customer == selected_customer]
    selected_cat = st.selectbox("Ангилал сонгох", 
                                 filtered_df[filtered_df['customer'] == selected_customer]['category'].unique())
    customer_row = fil_df[fil_df.category == selected_cat].iloc[0]
    #######HERE######################
    st.divider()
    metrics = customer_row['metric_array']
    scaled_score = customer_row['score']
    avg_score = customer_row['cat_avg_score']
    if scaled_score > avg_score:
        color = 'lightgreen'  # Above average
    else:
        color = '#f1807e'    # Below average
    st.header(f"Харилцагчийн дүн")
    st.markdown(f"<h1 style='color: {color};'>{scaled_score * 100:.2f}%</h1>", unsafe_allow_html=True)
    st.subheader("Харилцагчийн KPI үзүүлэлтүүд")
    cols = st.columns(len(metrics))
    metric_labels = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "GP"]
    avg_metrics = customer_row['avg_metric_array']
    for col, label, value, avg_value in zip(cols, metric_labels, metrics, avg_metrics):
        # Format the value as a whole number percentage
        whole_percentage = float(value)  # Convert to whole number
        avg_percentage = float(avg_value)  # Convert average to whole number

        # Determine arrow direction and color
        if whole_percentage > avg_percentage:
            arrow = "<span style='color: lightgreen;'>▲</span>"  # Green triangle for above average
        elif whole_percentage < avg_percentage:
            arrow = "<span style='color: #f1807e;'>▼</span>"  # Red triangle for below average
        else:
            arrow = "<span style='color: gray;'>•</span>"  # Gray circle for equal

        # Display each metric in a cell with a tinge of off-white
        col.markdown(
            f"""
            <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; text-align: center; font-size: 16px;">
                <strong>{label}</strong>: {whole_percentage:.2f}%
            </div>
            <div style="text-align: center; font-size: 16px;">
                {arrow}
            </div>
            """,
            unsafe_allow_html=True
        )
    st.divider()
    s_cf = fil_df[fil_df.category == selected_cat]
    s_cf = s_cf[['customer', 'category', 'divcnt', 'skucnt', 'purchasecnt', 'inqty',
   'orderqty', 'profit', 'avgcomiss', 'avgprice', 'inamt', 'intaxedamt',
   'errors_qty', 'errors_cnt', 'manualOrder', 'autoOrder', 'avgLeadTime','avg_error']]
    s_cf.columns = ['Харилцагч','Ангилал','Салбарын тоо',
                        'Барааны тоо','ХА тоо',
                        'ХА-аар орсон барааны тоо ширхэг','Захиалсан барааны тоо ширхэг',
                         'Нийт ашиг','Дундаж GP','Дундаж үнэ','Орлогын дүн','Татварын дараах орлогын дүн','Зөрүүтэй барааны тоо ширхэг','Зөрүүтэй ХА-ын тоо',
                         'Гараар орсон захиалга','Автомат захиалга','Ирэх хугацааны дундаж (хоног)','Зөрүүний дундаж']
    expander = st.expander("Харилцагчийн мэдээлэл харах", expanded=True)  # You can set expanded to True or False

    with expander:
        for i in s_cf.columns:
            st.write(f"<strong>{i}</strong>: {s_cf[i].iloc[0]}", unsafe_allow_html=True)

    customer_score = customer_row['score'] * 100
    avg_score = customer_row['cat_avg_score'] * 100
    category = customer_row['category']
    relative_scores_data = {
        'Хэмжигч': ['Харилцагчийн Оноо', 'Төрлийн Дундаж Оноо'],
        'Оноо': [customer_score, avg_score],
        'Төрөл': ['Харилцагч', 'Төрлийн дундаж']
    }
    relative_scores_df = pd.DataFrame(relative_scores_data)

    relative_scores_df['Оноо'] = relative_scores_df['Оноо'].round(2).astype('str') + '%'
    fig_relative_scores = px.bar(relative_scores_df, x='Хэмжигч', y='Оноо', color='Төрөл',
                                  color_discrete_sequence=['lightblue', 'gray'],
                                  title=f'Төрлийн харицуулалт: Төрөл {category}',
                                  text='Оноо')
    fig_relative_scores.update_traces(textfont_size=14, textposition='inside')
    st.plotly_chart(fig_relative_scores)


    metrics_labels = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "GP"]
    customer_metrics = customer_row['metric_array']
    avg_metrics = customer_row['avg_metric_array']

    data = {
        "Хэмжигч": metrics_labels * 2,  # Repeat labels for customer and average
        "Утга (%)": list(customer_metrics) + list(avg_metrics),  # Combine customer and avg metrics
        "Төрөл": ["Харилцагч"] * len(metrics_labels) + ["Бүлгийн дундаж"] * len(metrics_labels)  # Tag for grouping
    }

    bar_chart_df = pd.DataFrame(data)
    bar_chart_df['Утга (%)'] = bar_chart_df['Утга (%)']
    bar_chart_df['Утга (%)'] = bar_chart_df['Утга (%)'].round(2).astype('str') + '%'

    fig = px.bar(bar_chart_df, x='Хэмжигч', y='Утга (%)', color='Төрөл',
                 barmode='group', 
                 title=f'Төрлийн дундажтай харицуулсан KPI',
                 color_discrete_sequence=['lightblue', 'gray'],
                 text='Утга (%)')

    fig.update_traces(textfont_size=14, textposition='outside')
    st.plotly_chart(fig)


    fdf = filtered_df[filtered_df.category == customer_row['category']]
    # Create a new column to indicate if the customer is selected
    fdf['is_selected'] = fdf['customer'] == selected_customer

    # Create the scatter plot using Plotly Express
    fig = px.scatter(fdf, x='purchasecnt', y='profit', text='customer', title='Төрлийн задаргаа',
                     color_discrete_sequence=['gray'])  # Default color for all points

    # Highlight the selected customer
    selected_customer_data = fdf[fdf['customer'] == selected_customer]

    # Plot the selected customer's point with a different color
    if not selected_customer_data.empty:
        fig.add_trace(px.scatter(selected_customer_data, x='purchasecnt', y='profit', text='customer',
                                  color_discrete_sequence=['red']).data[0])

    # Add annotation to make it clear
    fig.add_annotation(
        x=selected_customer_data['purchasecnt'].values[0],
        y=selected_customer_data['profit'].values[0],
        text=selected_customer,
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30,
        font=dict(color='red')
    )

    # Customize layout
    fig.update_traces(textposition='top center')  # Position text above points
    fig.update_layout(xaxis_title='ХА тоо', yaxis_title='Ашиг')

    # Show the plot in Streamlit
    st.plotly_chart(fig)
    st.dataframe(fdf)                       
with st.expander("Тусгаарласан харилцагчид:"):
             st.write(filtered_out_customers)



    # with tab1:
    #     subprocess.run(["streamlit", "run", "homePage.py"], check=True)

    # with tab3:
    #     subprocess.run(["streamlit", "run", "clusterScore.py"], check=True)