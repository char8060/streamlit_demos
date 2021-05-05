import streamlit as st
import pandas as pd
import altair as alt
from churn_model import *

FILE_PATH = 'data.csv'

@st.cache(suppress_st_warning=True)
def get_data():
    infr_data=pd.read_csv(FILE_PATH)    
    st.write("Cache miss: expensive_computation ran")
    return infr_data

#data_load_state = st.text('Loading data...')
df = get_data()

model = ChurnModel()
model.read_pretrained(model_path='artifacts/v0.pkl')

#display sliders in sidebar
st.sidebar.write('Features')
feat1_sl = st.sidebar.slider('f1',min_value=-100.,max_value=5000.,value=df['feat1'].mean())
feat2_sl = st.sidebar.slider('f2',min_value=0.,max_value=2.,value=1.)
feat3_sl = st.sidebar.slider('f3',min_value=0.,max_value=200.,value=df['feat3'].mean())
feat4_sl = st.sidebar.slider('f4',min_value=0.,max_value=2.,value=1.)
feat5_sl = st.sidebar.slider('f5',min_value=0.,max_value=2.,value=1.)
feat6_sl = st.sidebar.slider('f6',min_value=0.,max_value=30.,value=df['feat6'].mean())
feat7_sl = st.sidebar.slider('f7',min_value=0.,max_value=2.,value=df['feat7'].mean())
feat8_sl = st.sidebar.slider('f8',min_value=0.,max_value=2.,value=1.)
feat9_sl = st.sidebar.slider('f9',min_value=0.,max_value=2.,value=1.)

xdf = pd.DataFrame({'mrr':[feat1_sl],
                    'mrr_lag1_ratio':[feat2_sl],
                    'age_months':[feat3_sl],
                    'storage_lag1_ratio':[feat4_sl],
                    'support_lag1_ratio':[feat5_sl],
                    'total_items':[feat6_sl],
                    'total_items_lag1_ratio': [feat7_sl],
                    'bandwidth_lag1_ratio':[feat8_sl],
                    'usage_lag1_ratio':[feat9_sl]
                    })

preds_df = model.predict(xdf)


st.write('Model Version',preds_df.model_version[0])
st.write('Churn Estimation',preds_df.churn_est[0])

data = df

base = alt.Chart(df[['target']])

chart2 = (
    base.mark_bar()\
    .encode(
        alt.X('target:Q',bin=alt.Bin(extent=[-0.5,0.5],maxbins=190)),
        y='count()'
    )
)

#responsive zoom
#https://altair-viz.github.io/gallery/histogram_responsive.html?highlight=histogram
#interactivce avg line
#https://altair-viz.github.io/gallery/selection_layer_bar_month.html

#constant value vertical line
rule = alt.Chart(preds_df).mark_rule(color='red').encode(
        x='churn_est:Q',
        size=alt.value(1)
    )


st.altair_chart(chart2+rule, use_container_width=True)
