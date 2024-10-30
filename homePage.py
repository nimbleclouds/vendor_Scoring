import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objs as go
import numpy as np
import hmac
import subprocess
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.decomposition import PCA
from streamlit_metrics import metric, metric_row

def check_password():
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if st.session_state.get("password_correct", False):
        return True
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Нууц үг буруу байна")
    return False


if not check_password():
     st.stop()


st.set_page_config(
    page_title="Nomin Supplier Dynamic Scoring",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSP-FdJah9dGxPHqBmdDi-hkkqO_QmlpVIWHg&s",
    layout="wide"
)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
st.image("https://erp15.nomin.mn/web/image/125495-0f10b8a0/%D0%9D%D0%BE%D0%BC%D0%B8%D0%BD%20%D0%9B%D0%BE%D0%B3%D0%BE.png", width=150)  # Adjust width as needed

tab_names = ["Үндсэн үзүүлэлтүүд", "Категорийн оноо", "Бүлгийн оноо"]
# Initialize session state for active tab if it doesn't exist
if "active_tab" not in st.session_state:
    st.session_state.active_tab = tab_names[0]  # Set default active tab

# Dropdown for tab selection
selected_tab = st.sidebar.selectbox("Сонгох таб:", tab_names, index=tab_names.index(st.session_state.active_tab))

# Update the active tab based on the dropdown selection
st.session_state.active_tab = selected_tab

# cols = st.columns(len(tab_names))
# for i,tab_name in enumerate(tab_names):
#     with cols[i]:
#         if st.button(tab_name):
#             st.session_state.active_tab = tab_name  # Set the active tab
        
if st.session_state.active_tab == tab_names[0]:
        st.markdown("<br>", unsafe_allow_html=True)

        part1 = pd.read_csv('data/orderData_category1.csv')
        part2 = pd.read_csv('data/orderData_category2.csv')
        part3 = pd.read_csv('data/orderData_category3.csv')
        part4 = pd.read_csv('data/orderData_category4.csv')
        odf = pd.concat([part1, part2, part3, part4], ignore_index=True)

        xpart1 = pd.read_csv('data/purchaseData_category1.csv')
        xpart2 = pd.read_csv('data/purchaseData_category2.csv')
        xpart3 = pd.read_csv('data/purchaseData_category3.csv')
        xpart4 = pd.read_csv('data/purchaseData_category4.csv')
        pdf = pd.concat([xpart1, xpart2, xpart3, xpart4], ignore_index=True)
    
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
        st.title("Үндсэн үзүүлэлтүүд")
        col_home_1, col_home_2 = st.columns(2)
        with col_home_1:
            category_profit = df.groupby('category')['profit'].sum().reset_index()
            category_profit = category_profit.sort_values(by='profit').tail(15)
            fig = px.bar(category_profit, 
                          x='profit', 
                          y='category', 
                          orientation='h', 
                          title='Хамгийн өндөр ашигтай 15 барааны ангилал',
                          labels={'profit': 'Ашиг', 'category': 'Барааны ангилал'}, text='profit')
            fig.update_layout(xaxis_tickprefix="", xaxis_tickformat=",", xaxis_title='Ашиг')
            st.plotly_chart(fig, use_container_width=True)
            st.empty()
            grouped_df = df.groupby('category').sum().reset_index()[['category', 'inamt']]
            top_categories = grouped_df.sort_values(['inamt']).tail(15)
            fig2 = px.bar(top_categories, 
                          x='inamt', 
                          y='category', 
                          orientation='h', 
                          labels={'inamt': 'Дүн', 'category': 'Барааны ангилал'},
                          title='Хамгийн өндөр борлуулалтын дүнтэй 15 барааны ангилал',
                          text='inamt')
            fig2.update_layout(xaxis_tickformat=',', xaxis_title='Дүн', yaxis_title='Барааны ангилал')
            st.plotly_chart(fig2, use_container_width=True)
            st.empty()
            grouped_df = df.groupby('customer').sum().reset_index()[['customer', 'profit']]
            top_customers = grouped_df.sort_values(['profit']).tail(15)
            fig3 = px.bar(top_customers, 
                          x='profit', 
                          y='customer', 
                          orientation='h', 
                          labels={'profit': 'Ашиг', 'customer': 'Харилцагч'},
                          title='Хамгийн өндөр ашигтай 15 харилцагч',
                          text='profit')
            fig3.update_layout(xaxis_tickformat=',', xaxis_title='Ашиг', yaxis_title='Харилцагч')
            st.plotly_chart(fig3, use_container_width=True)
        with col_home_2:
            grouped_df = df.groupby('customer').sum().reset_index()[['customer', 'inamt']]
            top_customers = grouped_df.sort_values(['inamt']).tail(15)
            fig4 = px.bar(top_customers, 
                          x='inamt', 
                          y='customer', 
                          orientation='h', 
                          labels={'inamt': 'Орлого дүн', 'customer': 'Харилцагч'},
                          title='Хамгийн өндөр орлогын дүн 15 харилцагч',
                          text='inamt')
            fig4.update_layout(xaxis_tickformat=',', xaxis_title='Орлого дүн', yaxis_title='Харилцагч')
            st.plotly_chart(fig4, use_container_width=True)
            st.empty()
            grouped_df = df.groupby('category').sum().reset_index()[['category', 'purchasecnt']]
            top_categories = grouped_df.sort_values(['purchasecnt']).tail(15)
            fig5 = px.bar(top_categories, 
                          x='purchasecnt', 
                          y='category', 
                          orientation='h', 
                          labels={'purchasecnt': 'ХА тоо', 'category': 'Барааны ангилал'},
                          title='Хамгийн өндөр худалдан авалтын тоотой 15 харилцагч',
                          text='purchasecnt')
            fig5.update_layout(xaxis_title='ХА тоо', yaxis_title='Барааны ангилал')
            st.plotly_chart(fig5, use_container_width=True)
            st.empty()
            grouped_df = df.groupby('category').sum().reset_index()[['category', 'inqty']]
            top_categories = grouped_df.sort_values(['inqty']).tail(15)
            fig6 = px.bar(top_categories, 
                          x='inqty', 
                          y='category', 
                          orientation='h', 
                          labels={'inqty': 'Оролтын тоо ширхэг', 'category': 'Барааны ангилал'},
                          title='Хамгийн өндөр оролтын тоо ширхэгтэй 15 барааны ангилал',
                          text='inqty')
            fig6.update_layout(xaxis_title='Оролтын тоо ширхэг', yaxis_title='Барааны ангилал')
            st.plotly_chart(fig6, use_container_width=True)
        category_stats = df.groupby('category').agg(
            total_profit=('profit', 'sum'),
            unique_customers=('customer', 'nunique')
        ).reset_index()
        top_categories = category_stats.nlargest(15, 'total_profit')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_categories['category'],
            y=top_categories['total_profit'],
            name='Нийт ашиг',
            marker_color='blue',
            opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=top_categories['category'],
            y=top_categories['unique_customers'],
            name='Харилцагч',
            mode='lines+markers',
            marker=dict(color='red'),
            yaxis='y2'  # Specify y-axis for unique customers
        ))
        fig.update_layout(
            title='Хамгийн өндөр ашигтай 15 барааны ангилал (харилцагчийн тоотой)',
            xaxis_title='Барааны ангилал',
            yaxis_title='Нийт ашиг',
            yaxis=dict(title='Нийт ашиг', side='left', titlefont=dict(color='blue'), tickformat=','),
            yaxis2=dict(overlaying='y', side='right', titlefont=dict(color='red'), tickformat=',', showgrid=False, title='', showline=False),
            showlegend=True,
            xaxis_tickangle=-45,
        )
        for i, row in top_categories.iterrows():
            fig.add_annotation(
                x=row['category'],
                y=row['unique_customers'],
                text=str(row['unique_customers']),
                showarrow=False,  # Hide arrows
                font=dict(color='red'),
                yshift=10  # Adjust position slightly above the point
            )
        st.plotly_chart(fig, use_container_width=True)
elif st.session_state.active_tab == tab_names[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        part1 = pd.read_csv('data/orderData_category1.csv')
        part2 = pd.read_csv('data/orderData_category2.csv')
        part3 = pd.read_csv('data/orderData_category3.csv')
        part4 = pd.read_csv('data/orderData_category4.csv')
        odf = pd.concat([part1, part2, part3, part4], ignore_index=True)

        xpart1 = pd.read_csv('data/purchaseData_category1.csv')
        xpart2 = pd.read_csv('data/purchaseData_category2.csv')
        xpart3 = pd.read_csv('data/purchaseData_category3.csv')
        xpart4 = pd.read_csv('data/purchaseData_category4.csv')
        pdf = pd.concat([xpart1, xpart2, xpart3, xpart4], ignore_index=True)
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
            st.divider()
            st.warning('Энэхүү барааны төрлүүд нь нэг болон түүнээс доош харилцагчтай тул үнэлэх боломжгүй.')
        df = df[['customer', 'category', 'divcnt', 'skucnt', 'purchasecnt', 'inqty',
               'orderqty', 'profit', 'avgcomiss', 'avgprice', 'inamt', 'intaxedamt',
               'errors_qty', 'errors_cnt', 'manualOrder', 'autoOrder', 'avgLeadTime',
               'avg_error']]
        metrics = ['profit', 'purchasecnt', 'orderqty', 'avgLeadTime', 'avg_error', 'divcnt', 'skucnt', 'avgcomiss']
        def remove_outliers_iqr_within_category(df, columns):
            filtered_out_customers = []  
            def filter_outliers(group):
                original_customers = group['customer'].tolist()
                for column in columns:
                    Q1 = group[column].quantile(0.25)
                    Q3 = group[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    group = group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
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
            'avgcomiss': "Дундаж Commission"
        }
        def scale_within_category(group):
            if len(group) > 1:
                scaled_values = scaler.fit_transform(group[metrics])
                return pd.DataFrame(scaled_values, columns=[f'minmax_{col}' for col in metrics], index=group.index)
            else:
                return pd.DataFrame(0, index=group.index, columns=[f'minmax_{col}' for col in metrics])
        st.title("Ханган нийлүүлэгчийн барааны төрлийн динамик үнэлгээ")
        st.sidebar.header("Жин тохируулах - Барааны төрөл")
        input_values = {}
        for key in weights.keys():
            input_values[key] = st.sidebar.number_input(
                f"{display_names[key]}(%)",
                min_value=-100.0,
                max_value=100.0,
                value=weights[key] * 100,  # Start with the initial weight
                step=0.1) / 100
        total_weight = sum(input_values.values())
        st.sidebar.write(f"Нийт дүн: {total_weight * 100:.2f}%")
        if total_weight > 1.0:
            st.sidebar.warning("Нийт жингийн дүн 100%-аас хэтэрсэн тул засна уу?")
        elif total_weight < 1.0:
            st.sidebar.warning("Нийт жингийн дүн 100%-аас бага байж болохгүй тул засна уу?")
        else:
            st.sidebar.success("Тохирсон.")
        if 'weights' not in st.session_state:
            st.session_state.weights = {}
        if 'selected_metrics' not in st.session_state:
            st.session_state.selected_metrics = []
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
            "Дундаж Commission": "avgcomiss",
        }
        if st.sidebar.button("Хадгалах"):
            for key in weights.keys():
                weights[key] = input_values[key]
            st.session_state.weights = weights
            st.sidebar.success("Амжилттай хадгалагдлаа!")
            st.sidebar.write(weights)
        st.markdown("IQR дундаж тооцоолох баганууд сонгох")
        metrics_mapping = {
            "Ашгийн хувь": "profit",
            "ХА тоо": "purchasecnt",
            "Захиалгын тоо": "orderqty",
            "Дундаж хүргэлтийн хугацаа (хоног)": "avgLeadTime",
            "Алдаатай захиалгын хувь": "avg_error",
            "Нийлүүлдэг салбаруудын тоо": "divcnt",
            "Нийлүүлдэг SKU тоо": "skucnt",
            "Дундаж Commission": "avgcomiss",
        }
        options = list(metrics_mapping.keys())
        valid_defaults = [item for item in st.session_state.selected_metrics if item in options]
        ms_arr = st.multiselect(
            "2 хэмжүүр сонгох",
            options,
            max_selections=2,
            default=valid_defaults  # Ensure defaults are valid options
        )
        selected_metrics = [metrics_mapping[item] for item in ms_arr if item in metrics_mapping]
        st.session_state.selected_metrics = selected_metrics  # Update session state
        st.success("Дундаж хязгаарлалт тодорхойлох үзүүлэлтүүд хадгалагдлаа: " + ", ".join(selected_metrics))
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
        avg_scores = filtered_df.groupby('category')['score'].mean().reset_index()
        avg_scores.rename(columns={'score': 'cat_avg_score'}, inplace=True)

        metric_array_df = pd.DataFrame(filtered_df['metric_array'].tolist())
        metric_array_df['category'] = filtered_df['category'].values
        avg_metric_array = metric_array_df.groupby('category').mean().reset_index()
        avg_metric_array['avg_metric_array'] = avg_metric_array.iloc[:, 1:].values.tolist()
        filtered_df = filtered_df.merge(avg_scores, on='category', how='left')
        filtered_df = filtered_df.merge(avg_metric_array[['category', 'avg_metric_array']], on='category', how='left')
        # Select category first
	selected_cat = st.selectbox("Ангилал сонгох", filtered_df['category'].unique())

	if selected_cat:
   		 # Filter customers based on the selected category
    		customers_in_category = filtered_df[filtered_df['category'] == selected_cat]['customer'].unique()
    		selected_customer = st.selectbox("customer", customers_in_category)
    		if selected_customer:
       			 st.session_state.selected_customer = selected_customer  # Store selected customer in session state
      		         fil_df = filtered_df[filtered_df.customer == selected_customer]
        
        customer_row = fil_df.iloc[0]  # Get the first row for the selected customer
        #######HERE######################
        st.divider()
            st.divider()
            metrics = customer_row['metric_array']
            scaled_score = customer_row['score']
            avg_score = customer_row['cat_avg_score']
            color = 'lightgreen' if scaled_score > avg_score else '#f1807e'
            st.header("Харилцагчийн дүн")
            st.markdown(f"<h1 style='color: {color};'>{scaled_score * 100:.2f}%</h1>", unsafe_allow_html=True)
            st.subheader("Харилцагчийн KPI үзүүлэлтүүд")
            cols = st.columns(len(metrics))
            metric_labels = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "Commission"]
            avg_metrics = customer_row['avg_metric_array']
            for col, label, value, avg_value in zip(cols, metric_labels, metrics, avg_metrics):
                whole_percentage = float(value)  # Convert to whole number
                avg_percentage = float(avg_value)  # Convert average to whole number
                delta = whole_percentage - avg_percentage
                delta_color = "normal" if delta == 0 else ("inverse" if delta < 0 else "normal")
                col.metric(label=label, value=f"{whole_percentage:.2f}%", delta=f"{delta:.2f}%", delta_color=delta_color)
            st.divider()
            s_cf = fil_df[fil_df.category == selected_cat]
            s_cf = s_cf[['customer', 'category', 'divcnt', 'skucnt', 'purchasecnt', 'inqty',
                          'orderqty', 'profit', 'avgcomiss', 'avgprice', 'inamt', 'intaxedamt',
                          'errors_qty', 'errors_cnt', 'manualOrder', 'autoOrder', 'avgLeadTime', 'avg_error']]
            s_cf.columns = ['Харилцагч', 'Ангилал', 'Салбарын тоо',
                             'Барааны тоо', 'ХА тоо',
                             'ХА-аар орсон барааны тоо ширхэг', 'Захиалсан барааны тоо ширхэг',
                             'Нийт ашиг', 'Дундаж Commission', 'Дундаж үнэ', 'Орлогын дүн', 'Татвар орсон орлогын дүн', 
                             'Зөрүүтэй барааны тоо ширхэг', 'Зөрүүтэй ХА-ын тоо',
                             'Гараар орсон захиалга', 'Автомат захиалга', 'Ирэх хугацааны дундаж (хоног)', 'Зөрүүний дундаж']

            expander = st.expander("Харилцагчийн мэдээлэл харах", expanded=True)  # You can set expanded to True or False
            with expander:
                for i in s_cf.columns:
                    value = s_cf[i].iloc[0]  # Get the value from the DataFrame
                    col = st.columns([1, 0.1, 1])  # Create three columns: label, separator, and value
                    with col[0]:  # Left column for label
                        st.markdown(f"<strong>{i}</strong>", unsafe_allow_html=True)
                    with col[1]:  # Middle column for separator
                        st.markdown("<div style='border-left: 2px solid gray; height: 30px; margin: 0 auto;'></div>", unsafe_allow_html=True)
                    with col[2]:  # Right column for value
                        st.markdown(f"<p style='font-size: 16px;'>{value}</p>", unsafe_allow_html=True)
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
            metrics_labels = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "Commission"]
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
            fdf['is_selected'] = fdf['customer'] == selected_customer
            fig = px.scatter(fdf, x='purchasecnt', y='profit', text='customer', title='Төрлийн задаргаа',
                             color_discrete_sequence=['gray'])  # Default color for all points
            selected_customer_data = fdf[fdf['customer'] == selected_customer]
            if not selected_customer_data.empty:
                fig.add_trace(px.scatter(selected_customer_data, x='purchasecnt', y='profit', text='customer',
                                          color_discrete_sequence=['red']).data[0])
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
            fig.update_traces(textposition='top center')  # Position text above points
            fig.update_layout(xaxis_title='ХА тоо', yaxis_title='Ашиг')
            st.plotly_chart(fig)
            fdf_mn =  fdf[['customer', 'category', 'divcnt', 'skucnt', 'purchasecnt', 'inqty',
           'orderqty', 'profit', 'avgcomiss', 'avgprice', 'inamt', 'intaxedamt',
           'errors_qty', 'errors_cnt', 'manualOrder', 'autoOrder', 'avgLeadTime','avg_error','score', 'is_selected']]
            fdf_mn.columns = ['Харилцагч','Ангилал','Салбарын тоо',
                                'Барааны тоо','ХА тоо',
                                'ХА-аар орсон барааны тоо ширхэг','Захиалсан барааны тоо ширхэг',
                                 'Нийт ашиг','Дундаж Commission','Дундаж үнэ','Орлогын дүн','Татвар орсон орлогын дүн','Зөрүүтэй барааны тоо ширхэг','Зөрүүтэй ХА-ын тоо',
                                 'Гараар орсон захиалга','Автомат захиалга','Ирэх хугацааны дундаж (хоног)','Зөрүүний дундаж','Оноо','Сонгогдсон']
            st.dataframe(fdf_mn)                       
        with st.expander("Тусгаарласан харилцагчид:"):
                     st.write(filtered_out_customers)


elif st.session_state.active_tab == tab_names[2]:
    st.markdown("<br>", unsafe_allow_html=True)
    odf_cluster = pd.read_csv('data/orderData_cluster.csv')
    pdf_cluster= pd.read_csv('data/purchaseData_cluster.csv')
    odf_cluster.columns = ['year',
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
    pdf_cluster.columns = ['year','day','customer','custid','divcnt','skucnt','inqty','outqty','avgprice','avgcomiss',
                  'avgtaxedprice','inamt','outamt','intaxedamt','outtaxedamt','orderqty','purchasecnt','correct','missing','surplus','purchaseonly']
    pdf_cluster['customer'] = pdf_cluster['customer'].astype('str')
    odf_cluster['customer'] = odf_cluster['customer'].astype('str')
    pdf_cluster= pdf_cluster[~pdf_cluster['customer'].str.contains('Номин')]
    odf_cluster = odf_cluster[~odf_cluster['customer'].str.contains('Номин')]
    pdf_cluster= pdf_cluster[~pdf_cluster['customer'].str.contains('НОМИН')]
    odf_cluster = odf_cluster[~odf_cluster['customer'].str.contains('НОМИН')]
    odf_cluster['date'] = pd.to_datetime(odf_cluster['year'].astype(str) + odf_cluster['day'].astype(str).str.zfill(3), format='%Y%j')
    pdf_cluster['date'] = pd.to_datetime(pdf_cluster['year'].astype(str) + pdf_cluster['day'].astype(str).str.zfill(3), format='%Y%j')
    odf_cluster = odf_cluster.dropna()
    pdf_cluster= pdf_cluster.dropna()
    pdf_cluster['errors_qty'] = pdf_cluster['missing'] + pdf_cluster['surplus'] 
    pdf_cluster['errors_cnt'] = (pdf_cluster['errors_qty'] > 0).astype(int)
    pdf_cluster['profit'] = pdf_cluster['avgcomiss']/100
    pdf_cluster['profit'] = pdf_cluster['intaxedamt'] * pdf_cluster['profit']
    order_data_cluster = odf_cluster.groupby(['customer']).agg({
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
    purchase_data_cluster = pdf_cluster.groupby(['customer']).agg({
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
    df_cluster = purchase_data_cluster.merge(order_data_cluster,on=['customer'],how='left')
    df_cluster = df_cluster.drop(columns = ['skuCnt','divCnt','outqty','avgPrice','orderQty','avgPrice','totalAmt','orderCnt','confirmed'])
    df_cluster['avg_error'] = df_cluster['errors_cnt'] / df_cluster['purchasecnt']
    df_cluster = df_cluster.fillna(value=0)
    X = df_cluster.drop(columns=['customer'])  # Drop customer name
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)
    pca = PCA(n_components=8)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(8)])
    pca_df['customer'] = df_cluster['customer'] 
    pca_df['zscore_PC1'] = zscore(pca_df['PC1'])
    pca_df['zscore_PC2'] = zscore(pca_df['PC2'])
    threshold = 3
    outliers_cluster = pca_df[(abs(pca_df['zscore_PC1']) > threshold) | (abs(pca_df['zscore_PC2']) > threshold)]
    outlier_customers_cluster = outliers_cluster['customer'].unique()
    filtered_df_cluster = df_cluster[~df_cluster['customer'].isin(outlier_customers_cluster)]
    X = filtered_df_cluster.drop(columns=['customer']) 
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(4)])
    pca_df['customer'] = filtered_df_cluster['customer']
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    X = filtered_df_cluster.drop(columns=['customer'])
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_pca)
    filtered_df_cluster['Cluster'] = kmeans.labels_
    
    filtered_df_cluster['profit'] = filtered_df_cluster['profit'].astype('int64')
    if 'weights_cluster' not in st.session_state:
        st.session_state.weights_cluster = {
            'profit': 0.4,
            'purchasecnt': 0.25,
            'orderqty': 0.1,
            'avgLeadTime': 0.05,
            'avg_error': -0.1,
            'divcnt': 0.025,
            'skucnt': 0.025,
            'avgcomiss': 0.05
        }
    weights_cluster = st.session_state.weights_cluster
    display_names_cluster = {
        'profit': "Ашгийн хувь",
        'purchasecnt': "ХА тоо",
        'orderqty': "Захиалгын тоо",
        'avgLeadTime': "Дундаж хүргэлтийн хугацаа (хоног)",
        'avg_error': "Алдаатай захиалгын хувь",
        'divcnt': "Нийлүүлдэг салбаруудын тоо",
        'skucnt': "Нийлүүлдэг SKU тоо",
        'avgcomiss': "Дундаж Commission"
    }
    st.title("Ханган нийлүүлэгчийн бүлгийн динамик үнэлгээ")
    st.sidebar.header("Жин тохируулах - Бүлэг")
    input_values_cluster = {}
    for key in weights_cluster.keys():
        input_values_cluster[key] = st.sidebar.number_input(
            f"{display_names_cluster[key]}(%)",
            min_value=-100.0,
            max_value=100.0,
            value=weights_cluster[key] * 100,  # Start with the initial weight
            step=0.1,key = f"{key}+a") / 100
    total_weight_cluster = sum(input_values_cluster.values())
    st.sidebar.write(f"Нийт дүн: {total_weight_cluster * 100:.2f}%")
    if total_weight_cluster > 1.0:
        st.sidebar.warning("Нийт жингийн дүн 100%-аас хэтэрсэн тул засна уу?")
    elif total_weight_cluster < 1.0:
        st.sidebar.warning("Нийт жингийн дүн 100%-аас бага байж болохгүй тул засна уу?")
    else:
        st.sidebar.success("Тохирсон.")
    if st.sidebar.button("Хадгалах", key='Cluster_save'):
        for key in weights_cluster.keys():
            weights_cluster[key] = input_values_cluster[key]
        st.session_state.weights_cluster = weights_cluster
        st.sidebar.success("Амжилттай хадгалагдлаа!")
        st.sidebar.write(weights_cluster)
    if 'results_cluster' not in st.session_state:
        st.session_state.results_cluster = []
    if st.button("Тооцоолох"):
        results_cluster = []
        for cluster, group in filtered_df_cluster.groupby('Cluster'):
            scaler_cluster = MinMaxScaler()
            scaled_metrics_cluster = scaler_cluster.fit_transform(group[list(weights_cluster.keys())])
            scaled_df_cluster = pd.DataFrame(scaled_metrics_cluster, columns=weights_cluster.keys(), index=group.index)
            scaled_df_cluster['score'] = sum(scaled_df_cluster[col] * weight for col, weight in weights_cluster.items())
            scaled_df_cluster['scaled_score'] = (scaled_df_cluster['score'] - scaled_df_cluster['score'].min()) / (scaled_df_cluster['score'].max() - scaled_df_cluster['score'].min()) * 100
            total_score_cluster = scaled_df_cluster['score'].sum()
            for col in weights_cluster.keys():
                scaled_df_cluster[f'{col}_percent'] = (scaled_df_cluster[col] * weights_cluster[col]) / total_score_cluster * 100
            leader_idx_cluster = scaled_df_cluster['scaled_score'].idxmax()
            leader_metrics_cluster = scaled_df_cluster.loc[leader_idx_cluster, [f'{col}_percent' for col in ['profit', 'purchasecnt', 'orderqty', 'avgLeadTime', 'avg_error', 'divcnt', 'skucnt', 'avgcomiss']]]
            leader_score_cluster = scaled_df_cluster.loc[leader_idx_cluster, 'scaled_score']  # Get the leader's scaled score
            avg_metrics_cluster = scaled_df_cluster[[f'{col}_percent' for col in weights_cluster.keys()]].mean().tolist()
            avg_score_cluster = scaled_df_cluster['scaled_score'].mean()
            for idx, row in scaled_df_cluster.iterrows():
                results_cluster.append({
                    'customer': group.loc[idx, 'customer'],
                    'cluster': cluster,
                    'scaled_score': row['scaled_score'],
                    'metrics_percentages': [row[f'{col}_percent'] for col in ['profit', 'purchasecnt', 'orderqty', 'avgLeadTime', 'avg_error', 'divcnt', 'skucnt', 'avgcomiss']],
                    'cluster_leader': leader_metrics_cluster.tolist(),  # Add the leader's metrics percentages
                    'cluster_leader_score': leader_score_cluster,  # Add the leader's scaled score
                    'metrics_percentages_avg': avg_metrics_cluster,  # Add average metrics percentages
                    'avg_score': avg_score_cluster  # Add average scaled score
                })

        st.session_state.results_cluster = results_cluster
    if st.session_state.results_cluster:
        final_df_cluster = pd.DataFrame(st.session_state.results_cluster)
        selected_customer_cluster = st.selectbox("Харилцагч сонгох", final_df_cluster['customer'].unique())
        st.divider()
        if selected_customer_cluster:
            customer_row_cluster = final_df_cluster[final_df_cluster['customer'] == selected_customer_cluster].iloc[0]
            metrics_cluster = customer_row_cluster['metrics_percentages']
            scaled_score_cluster = customer_row_cluster['scaled_score']
            avg_score_cluster = customer_row_cluster['avg_score']
            if scaled_score_cluster > avg_score_cluster:
                color_cluster = 'lightgreen'  # Above average
            else:
                color_cluster = '#f1807e'    # Below average
            st.header(f"Харилцагчийн дүн")
            st.markdown(f"<h1 style='color: {color_cluster};'>{scaled_score_cluster:.2f}%</h1>", unsafe_allow_html=True)
            st.subheader("Харилцагчийн KPI үзүүлэлтүүд")
            cols_cluster = st.columns(len(metrics_cluster))
            metric_labels_cluster = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "Commission"]
            avg_metrics_cluster = customer_row_cluster['metrics_percentages_avg']
            
            for col, label, value, avg_value in zip(cols_cluster, metric_labels_cluster, metrics_cluster, avg_metrics_cluster):
                whole_percentage = float(value) * 100  # Convert to whole number
                avg_percentage = float(avg_value) * 100  # Convert average to whole number
                delta = whole_percentage - avg_percentage
            
                # Update delta_color logic
                if delta > 0:
                    delta_color = "normal"  # Positive difference
                elif delta < 0:
                    delta_color = "inverse"  # Negative difference
                else:
                    delta_color = "normal"  # No change can be treated as neutral
            
                # Set the metric with the appropriate delta color
                col.metric(label=label, value=f"{whole_percentage:.2f}%", delta=f"{delta:.2f}%", delta_color=delta_color)
            
            st.divider()
            
            
            s_cf_cluster = filtered_df_cluster[filtered_df_cluster.customer == selected_customer_cluster]
            s_cf_cluster.columns = ['Харилцагч','Салбарын тоо',
                                'Барааны тоо','ХА тоо',
                                'ХА-аар орсон барааны тоо ширхэг','Захиалсан барааны тоо ширхэг',
                                 'Нийт ашиг','Дундаж Commission','Дундаж үнэ','Орлогын дүн','Татварын дараах орлогын дүн','Зөрүүтэй барааны тоо ширхэг','Зөрүүтэй ХА-ын тоо',
                                 'Гараар орсон захиалга','Автомат захиалга','Ирэх хугацааны дундаж (хоног)','Зөрүүний дундаж','Хамрагдах бүлэг'
                                ]
            expander_cluster = st.expander("Харилцагчийн мэдээлэл харах", expanded=True)  # You can set expanded to True or False
            with expander_cluster:
                for i in s_cf_cluster.columns:
                    value_cluster = s_cf_cluster[i].iloc[0]  # Get the value from the DataFrame
                    col_cluster = st.columns([1, 0.1, 1])  # Create three columns: label, separator, and value
                    with col_cluster[0]:  # Left column for label
                        st.markdown(f"<strong>{i}</strong>", unsafe_allow_html=True)
                    with col_cluster[1]:  # Middle column for separator
                        st.markdown("<div style='border-left: 2px solid gray; height: 30px; margin: 0 auto;'></div>", unsafe_allow_html=True)
                    with col_cluster[2]:  # Right column for value
                        st.markdown(f"<p style='font-size: 16px;'>{value_cluster}</p>", unsafe_allow_html=True)
                        
                        

            customer_score_cluster = customer_row_cluster['scaled_score']
            avg_score_cluster = customer_row_cluster['avg_score']
            cluster_cluster = customer_row_cluster['cluster']
            relative_scores_data_cluster = {
                'Хэмжигч': ['Харилцагчийн Оноо', 'Бүлгийн Дундаж Оноо'],
                'Оноо': [customer_score_cluster, avg_score_cluster],
                'Төрөл': ['Харилцагч', 'Бүлгийн дундаж']
            }
            relative_scores_df_cluster = pd.DataFrame(relative_scores_data_cluster)
            relative_scores_df_cluster['Оноо'] = relative_scores_df_cluster['Оноо'].round(2).astype('str') + '%'
            fig_relative_scores_cluster = px.bar(relative_scores_df_cluster, x='Хэмжигч', y='Оноо', color='Төрөл',
                                          color_discrete_sequence=['lightblue', 'gray'],
                                          title=f'Бүлгийн харицуулалт: Бүлэг {cluster_cluster}',
                                          text='Оноо')
            fig_relative_scores_cluster.update_traces(textfont_size=14, textposition='inside')
            st.plotly_chart(fig_relative_scores_cluster)
            metrics_labels_cluster = ["Ашиг", "ХА", "PO", "Хугацаа", "Алдаа", "Салбар", "SKU", "Commission"]
            customer_metrics_cluster = customer_row_cluster['metrics_percentages']
            avg_metrics_cluster = customer_row_cluster['metrics_percentages_avg']
            data_cluster = {
                "Хэмжигч": metrics_labels_cluster * 2,  # Repeat labels for customer and average
                "Утга (%)": list(customer_metrics_cluster) + list(avg_metrics_cluster),  # Combine customer and avg metrics
                "Төрөл": ["Харилцагч"] * len(metrics_labels_cluster) + ["Бүлгийн дундаж"] * len(metrics_labels_cluster)  # Tag for grouping
            }
            bar_chart_df_cluster = pd.DataFrame(data_cluster)
            bar_chart_df_cluster['Утга (%)'] = bar_chart_df_cluster['Утга (%)'] * 100
            bar_chart_df_cluster['Утга (%)'] = bar_chart_df_cluster['Утга (%)'].round(2).astype('str') + '%'
            fig_cluster = px.bar(bar_chart_df_cluster, x='Хэмжигч', y='Утга (%)', color='Төрөл',
                         barmode='group', 
                         title=f'Бүлгийн дундажтай харицуулсан KPI',
                         color_discrete_sequence=['lightblue', 'gray'],
                         text='Утга (%)')
            fig_cluster.update_traces(textfont_size=14, textposition='outside')
            st.plotly_chart(fig_cluster)
            
            c_pca_df_cluster = pca_df.copy()
            c_pca_df_cluster = c_pca_df_cluster.merge(filtered_df_cluster[['customer','Cluster']], how='left', on='customer')
            c_pca_df_cluster = c_pca_df_cluster[c_pca_df_cluster.Cluster == customer_row_cluster.cluster]
            c_pca_df_cluster['is_selected'] = c_pca_df_cluster['customer'] == selected_customer_cluster
            fig_cluster_cluster = px.scatter(c_pca_df_cluster, x='PC1', y='PC2', text='customer', title='Бүлгийн задаргаа',
                             color_discrete_sequence=['gray'])  # Default color for all points
            selected_customer_data_cluster = c_pca_df_cluster[c_pca_df_cluster['customer'] == selected_customer_cluster]
            if not selected_customer_data_cluster.empty:
                fig_cluster_cluster.add_trace(px.scatter(selected_customer_data_cluster, x='PC1', y='PC2', text='customer',
                                          color_discrete_sequence=['red']).data[0])
            fig_cluster_cluster.add_annotation(
                x=selected_customer_data_cluster['PC1'].values[0],
                y=selected_customer_data_cluster['PC2'].values[0],
                text=selected_customer_cluster,
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                font=dict(color='red')
            )
            fig_cluster_cluster.update_traces(textposition='top center')  # Position text above points
            fig_cluster_cluster.update_layout(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2')
            st.plotly_chart(fig_cluster_cluster)
            
            fdf_mn_cluster = final_df_cluster[final_df_cluster.cluster == customer_row_cluster.cluster]
            f_Clust = filtered_df_cluster[filtered_df_cluster.Cluster == customer_row_cluster.cluster]
            f_Clust = f_Clust.merge(fdf_mn_cluster[['customer','scaled_score']],how='left',on='customer')
            f_Clust.columns = ['Харилцагч','Салбарын тоо',
                                'Барааны тоо','ХА тоо',
                                'ХА-аар орсон барааны тоо ширхэг','Захиалсан барааны тоо ширхэг',
                                 'Нийт ашиг','Дундаж Commission','Дундаж үнэ','Орлогын дүн','Татварын дараах орлогын дүн','Зөрүүтэй барааны тоо ширхэг','Зөрүүтэй ХА-ын тоо',
                                 'Гараар орсон захиалга','Автомат захиалга','Ирэх хугацааны дундаж (хоног)','Зөрүүний дундаж','Хамрагдах бүлэг','Оноо'
                                ]
            st.write(f_Clust)   
            
                
        with st.expander("Тусгаарласан харилцагчид"):
            for i in outlier_customers_cluster:
                st.markdown(f"- {i}")
    
