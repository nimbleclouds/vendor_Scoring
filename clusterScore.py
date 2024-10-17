import pandas as pd
import plotly.express as px
import streamlit as st

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

def run():

    st.image("https://erp15.nomin.mn/web/image/125495-0f10b8a0/%D0%9D%D0%BE%D0%BC%D0%B8%D0%BD%20%D0%9B%D0%BE%D0%B3%D0%BE.png", width=150)  # Adjust width as needed
    st.markdown("<br>", unsafe_allow_html=True)
    # tab1, tab2, tab3 = st.tabs(["Үндсэн үзүүлэлтүүд", "Категорийн оноо", "Кластерын оноо"])
    # with tab3:
    odf = pd.read_csv('data/orderData_cluster.csv')
    pdf = pd.read_csv('data/purchaseData_cluster.csv')

    odf.columns = ['year',
    'day',
    'customer',
    'custid',
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

    pdf.columns = ['year','day','customer','custid','divcnt','skucnt','inqty','outqty','avgprice','avgcomiss',
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
    order_data = odf.groupby(['customer']).agg({
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
    purchase_data = pdf.groupby(['customer']).agg({
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
    df = purchase_data.merge(order_data,on=['customer'],how='left')
    df = df.drop(columns = ['skuCnt','divCnt','outqty','avgPrice','orderQty','avgPrice','totalAmt','orderCnt','confirmed'])
    df['avg_error'] = df['errors_cnt'] / df['purchasecnt']
    df = df.fillna(value=0)
    X = df.drop(columns=['customer'])  # Drop customer name
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(8)])
    pca_df['customer'] = df['customer'] 
    pca_df['zscore_PC1'] = zscore(pca_df['PC1'])
    pca_df['zscore_PC2'] = zscore(pca_df['PC2'])
    threshold = 3
    outliers = pca_df[(abs(pca_df['zscore_PC1']) > threshold) | (abs(pca_df['zscore_PC2']) > threshold)]
    outlier_customers = outliers['customer'].unique()




    filtered_df = df[~df['customer'].isin(outlier_customers)]
    X = filtered_df.drop(columns=['customer']) 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(4)])
    pca_df['customer'] = filtered_df['customer']
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)

    X = filtered_df.drop(columns=['customer'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_pca)
    filtered_df['Cluster'] = kmeans.labels_
    centroids = filtered_df.groupby('Cluster').mean()
    filtered_df['profit'] = filtered_df['profit'].astype('int64')



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
    # Streamlit app title
    st.title("Ханган нийлүүлэгчийн бүлгийн динамик үнэлгээ")
    # Display the icon at the top left of the main page
    st.sidebar.header("Жин тохируулах")

    # Create a dictionary to hold the input values
    input_values = {}

    # Create input fields for each weight
    for key in weights.keys():
        input_values[key] = st.sidebar.number_input(
            f"{display_names[key]}(%)",
            min_value=-100.0,
            max_value=100.0,
            value=weights[key] * 100,  # Start with the initial weight
            step=0.1
        , key='cluster_input') / 100

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

    # Save weights button
    if st.sidebar.button("Хадгалах"):
        for key in weights.keys():
            weights[key] = input_values[key]
        st.session_state.weights = weights
        st.sidebar.success("Амжилттай хадгалагдлаа!")
        st.sidebar.write(weights)

    if 'results' not in st.session_state:
        st.session_state.results = []

    # Your existing logic for results and customer selection remains unchanged...

    if st.button("Тооцоолох"):
        # Initialize an empty list to collect results
        results = []

        # Group by cluster
        for cluster, group in filtered_df.groupby('Cluster'):
            # Apply MinMaxScaler
            scaler = MinMaxScaler()
            scaled_metrics = scaler.fit_transform(group[list(weights.keys())])

            # Create a DataFrame for the scaled metrics
            scaled_df = pd.DataFrame(scaled_metrics, columns=weights.keys(), index=group.index)

            # Calculate the weighted score
            scaled_df['score'] = sum(scaled_df[col] * weight for col, weight in weights.items())

            # Scale the score to 0-100
            scaled_df['scaled_score'] = (scaled_df['score'] - scaled_df['score'].min()) / (scaled_df['score'].max() - scaled_df['score'].min()) * 100

            # Calculate percentage contributions of each metric
            total_score = scaled_df['score'].sum()
            for col in weights.keys():
                scaled_df[f'{col}_percent'] = (scaled_df[col] * weights[col]) / total_score * 100

            # Find the leader for the current cluster
            leader_idx = scaled_df['scaled_score'].idxmax()
            leader_metrics = scaled_df.loc[leader_idx, [f'{col}_percent' for col in ['profit', 'purchasecnt', 'orderqty', 'avgLeadTime', 'avg_error', 'divcnt', 'skucnt', 'avgcomiss']]]
            leader_score = scaled_df.loc[leader_idx, 'scaled_score']  # Get the leader's scaled score

            # Calculate average metrics percentages and average score for the cluster
            avg_metrics = scaled_df[[f'{col}_percent' for col in weights.keys()]].mean().tolist()
            avg_score = scaled_df['scaled_score'].mean()

            # Collect the results
            for idx, row in scaled_df.iterrows():
                results.append({
                    'customer': group.loc[idx, 'customer'],
                    'cluster': cluster,
                    'scaled_score': row['scaled_score'],
                    'metrics_percentages': [row[f'{col}_percent'] for col in ['profit', 'purchasecnt', 'orderqty', 'avgLeadTime', 'avg_error', 'divcnt', 'skucnt', 'avgcomiss']],
                    'cluster_leader': leader_metrics.tolist(),  # Add the leader's metrics percentages
                    'cluster_leader_score': leader_score,  # Add the leader's scaled score
                    'metrics_percentages_avg': avg_metrics,  # Add average metrics percentages
                    'avg_score': avg_score  # Add average scaled score
                })

        st.session_state.results = results
    if st.session_state.results:
        final_df = pd.DataFrame(st.session_state.results)
        selected_customer = st.selectbox("Харилцагч сонгох", final_df['customer'].unique())
        st.divider()
        if selected_customer:
            customer_row = final_df[final_df['customer'] == selected_customer].iloc[0]
            metrics = customer_row['metrics_percentages']
            scaled_score = customer_row['scaled_score']
            avg_score = customer_row['avg_score']
            if scaled_score > avg_score:
                color = 'lightgreen'  # Above average
            else:
                color = '#f1807e'    # Below average
            st.header(f"Харилцагчийн дүн")
            st.markdown(f"<h1 style='color: {color};'>{scaled_score:.2f}%</h1>", unsafe_allow_html=True)
            st.subheader("Харилцагчийн KPI үзүүлэлтүүд")
            cols = st.columns(len(metrics))
            metric_labels = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "GP"]
            avg_metrics = customer_row['metrics_percentages_avg']

            for col, label, value, avg_value in zip(cols, metric_labels, metrics, avg_metrics):
                # Format the value as a whole number percentage
                whole_percentage = float(value) * 100  # Convert to whole number
                avg_percentage = float(avg_value) * 100  # Convert average to whole number

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

            s_cf = filtered_df[filtered_df.customer == selected_customer]
            s_cf.columns = ['Харилцагч','Салбарын тоо',
                                'Барааны тоо','ХА тоо',
                                'ХА-аар орсон барааны тоо ширхэг','Захиалсан барааны тоо ширхэг',
                                 'Нийт ашиг','Дундаж GP','Дундаж үнэ','Орлогын дүн','Татварын дараах орлогын дүн','Зөрүүтэй барааны тоо ширхэг','Зөрүүтэй ХА-ын тоо',
                                 'Гараар орсон захиалга','Автомат захиалга','Ирэх хугацааны дундаж (хоног)','Зөрүүний дундаж','Хамрагдах бүлэг'
                                ]
            # Assuming s_cf is your DataFrame
            expander = st.expander("Харилцагчийн мэдээлэл харах", expanded=True)  # You can set expanded to True or False

            with expander:
                for i in s_cf.columns:
                    st.write(f"<strong>{i}</strong>: {s_cf[i].iloc[0]}", unsafe_allow_html=True)

            customer_score = customer_row['scaled_score']
            avg_score = customer_row['avg_score']
            cluster = customer_row['cluster']
            relative_scores_data = {
                'Хэмжигч': ['Харилцагчийн Оноо', 'Бүлгийн Дундаж Оноо'],
                'Оноо': [customer_score, avg_score],
                'Төрөл': ['Харилцагч', 'Бүлгийн дундаж']
            }
            relative_scores_df = pd.DataFrame(relative_scores_data)
            relative_scores_df['Оноо'] = relative_scores_df['Оноо'].round(2).astype('str') + '%'
            fig_relative_scores = px.bar(relative_scores_df, x='Хэмжигч', y='Оноо', color='Төрөл',
                                          color_discrete_sequence=['lightblue', 'gray'],
                                          title=f'Бүлгийн харицуулалт: Бүлэг {cluster}',
                                          text='Оноо')
            fig_relative_scores.update_traces(textfont_size=14, textposition='inside')
            st.plotly_chart(fig_relative_scores)

            metrics_labels = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "GP"]
            customer_metrics = customer_row['metrics_percentages']
            avg_metrics = customer_row['metrics_percentages_avg']

            data = {
                "Хэмжигч": metrics_labels * 2,  # Repeat labels for customer and average
                "Утга (%)": list(customer_metrics) + list(avg_metrics),  # Combine customer and avg metrics
                "Төрөл": ["Харилцагч"] * len(metrics_labels) + ["Бүлгийн дундаж"] * len(metrics_labels)  # Tag for grouping
            }

            bar_chart_df = pd.DataFrame(data)
            bar_chart_df['Утга (%)'] = bar_chart_df['Утга (%)'] * 100
            bar_chart_df['Утга (%)'] = bar_chart_df['Утга (%)'].round(2).astype('str') + '%'

            fig = px.bar(bar_chart_df, x='Хэмжигч', y='Утга (%)', color='Төрөл',
                         barmode='group', 
                         title=f'Бүлгийн дундажтай харицуулсан KPI',
                         color_discrete_sequence=['lightblue', 'gray'],
                         text='Утга (%)')

            fig.update_traces(textfont_size=14, textposition='outside')
            st.plotly_chart(fig)

            c_pca_df = pca_df.copy()
            c_pca_df = c_pca_df.merge(filtered_df[['customer','Cluster']], how='left', on='customer')
            c_pca_df = c_pca_df[c_pca_df.Cluster == customer_row.cluster]
            # Create a new column to indicate if the customer is selected
            c_pca_df['is_selected'] = c_pca_df['customer'] == selected_customer

            # Create the scatter plot using Plotly Express
            fig = px.scatter(c_pca_df, x='PC1', y='PC2', text='customer', title='Бүлгийн задаргаа',
                             color_discrete_sequence=['gray'])  # Default color for all points

            # Highlight the selected customer
            selected_customer_data = c_pca_df[c_pca_df['customer'] == selected_customer]

            # Plot the selected customer's point with a different color
            if not selected_customer_data.empty:
                fig.add_trace(px.scatter(selected_customer_data, x='PC1', y='PC2', text='customer',
                                          color_discrete_sequence=['red']).data[0])

            # Add annotation to make it clear
            fig.add_annotation(
                x=selected_customer_data['PC1'].values[0],
                y=selected_customer_data['PC2'].values[0],
                text=selected_customer,
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                font=dict(color='red')
            )

            # Customize layout
            fig.update_traces(textposition='top center')  # Position text above points
            fig.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')

            # Show the plot in Streamlit
            st.plotly_chart(fig)


    # Expander for outlier customers
    with st.expander("Тусгаарласан харилцагчид"):
        for i in outlier_customers:
            st.markdown(f"- {i}")

    # with tab1:
    #     subprocess.run(["streamlit", "run", "homePage.py"], check=True)     
    # with tab2:
    #     subprocess.run(["streamlit", "run", "categoryScore.py"], check=True)

