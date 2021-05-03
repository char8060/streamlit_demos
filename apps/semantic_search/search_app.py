import sys
import os
from datetime import datetime

import yaml
import base64

from sentence_transformers import SentenceTransformer, util
from scipy import spatial
import streamlit as st
import pandas as pd
import altair as alt


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def cos_sim(v1,v2):
    return 1 - spatial.distance.cosine(v1, v2)

def query(input_str,df,model,thresh=0.75):
    result = df  
    query_embedding = model.encode(input_str.lower()
                                            , show_progress_bar=True)
        
    #print('running query...')
    result['cos_sim'] = result['embedding'].apply(lambda x: cos_sim(x,query_embedding))
    
    #return full dataset similarities for plotting
    plot_df = result[['cos_sim']]
    
    result = result[result['cos_sim'] >= thresh ]
    result = result[['ticket_number','cos_sim','ticket_subject','ticket_created_method','first_public_response_queue','msg','ticket_start_datetime']]
    result.sort_values(by='cos_sim', ascending=False,inplace=True)
    return result, plot_df

#===========================================#
#        Loads Model and word_to_id         #
#===========================================#

@st.cache(suppress_st_warning=True, 
          allow_output_mutation=True)
def load_data():
    CONFIG = '../model/config.yaml'

    with open(CONFIG,'r') as file:
        cfg = yaml.full_load(file)
        print('reading cfg')
        
    print('loading pickle file...')
    df = pd.read_pickle(cfg['embeddings_file'])
    print('done loading data')
    #load model
    #base = 'roberta-large-nli-stsb-mean-tokens'#'distilbert-base-nli-mean-tokens'
    print('loading transformer....')
    base = '../sbert.net_models_roberta-large-nli-stsb-mean-tokens'
    model = SentenceTransformer(base)
    print('ran expensive loads')
    return df, model
    

df, model = load_data()
#===========================================#
#              Streamlit Code               #
#===========================================#

st.title('CORE Semantic Search')
#st.write(desc)

user_input = st.text_input('Enter query for ticket subjects')
thresh_sl = st.slider('search results threshold',min_value=-1.,max_value=1.,value=0.75)

with open('streamlit_weblog.txt', 'a') as the_file:
    the_file.write(f'{user_input}, {datetime.now()}\n')

if st.button('Search'):
    #run query
    temp = df[['embedding','ticket_number','ticket_created_method','ticket_subject','msg','first_public_response_queue','ticket_start_datetime']]
    with st.spinner('Running query....'):
        df_out,plot_df = query(user_input,temp,model,thresh=thresh_sl)

    st.markdown("## Search Results")
    st.markdown(f"{df_out.shape[0]} matching tickets")
    st.dataframe(df_out[['ticket_number','cos_sim','ticket_subject','ticket_created_method','msg','first_public_response_queue']]) # will display the dataframe
    #st.table(datatable)# will display the table
    
    tmp_download_link = download_link(df_out, 'data.csv', 'Click here to download your data')
    st.markdown(tmp_download_link, unsafe_allow_html=True)
    
    st.markdown("## Similarity Distribution")
    base = alt.Chart(plot_df[['cos_sim']])
    chart2 = (
        base.mark_bar()\
        .encode(
            alt.X('cos_sim:Q',bin=alt.Bin(extent=[-0.5,1.],maxbins=100)),
            y='count()'
        )
    )
    

    thresh_df = pd.DataFrame({'thresh':[thresh_sl]})
    rule = alt.Chart(thresh_df).mark_rule(color='red').encode(
        x='thresh:Q',
        size=alt.value(1)
    )
    st.altair_chart(chart2+rule, use_container_width=True)
        
    st.markdown("## Results Trend")
    c = alt.Chart(df_out).mark_bar().encode(
        x='yearmonth(ticket_start_datetime):O',
        y='count():Q',
        color='ticket_created_method:N'
    )
    st.altair_chart(c,use_container_width=True)
