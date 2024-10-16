import streamlit as st
import data
import plots
import dicts
import pandas as pd
from dictionaries import met_sites
from wind_rose import wind_rose
from datetime import datetime
from summary_plot import summary_plot
from calendar_plot import calendar
from polar_frequency import polar_frequency_plot
from polar_plot import polar_plot
from map_sites import map_sites
from polar_cluster import polar_cluster
from time_plot import time_plot
from theil_sen_plot import theil_sen_plot

def main():
    st.set_page_config(page_title='Air Quality Python Tools Demo App', layout='wide')
    st.title('Air Quality Python Tools Demo App')
    aq_sources = ["aurn", "saqn", "aqe", "waqn", "ni"]

    which_source = st.selectbox('Select AQ Source', options = aq_sources)
    
    metadata = data.import_aq_meta(which_source)
    #st.dataframe(metadata)
    metadata = metadata[metadata['end_date'] == 'ongoing']
    
    site = st.selectbox('Select Site', options = sorted(metadata['site_id']))
    current_year = datetime.now().year
    c1, c2 = st.columns(2)
    with c1:
        start_year = st.number_input('Start Year', min_value = 2017, max_value = current_year, value = current_year-1)
    with c2:
        end_year = st.number_input('End Year', min_value = 2017, max_value = current_year, value = current_year)
    
    df = data.download_aurn_data(site, start_year, end_year, which_source)
    
    
    with st.expander('Raw Data'):
        st.dataframe(df, use_container_width=True)

    ####### Add new functions from here
    st.header('Plots')
    st.subheader('Theil-Sen Regression')
    t_ts = st.text_input('Select Time Average Strategy (leave blank for none)', value=None, key='ts_tav')
    col_ts = st.selectbox('Select Column for Theil-Sen regression', options=sorted(df.keys()))
    fig = theil_sen_plot(df, pollutant_col=col_ts, agg_freq=t_ts)
    st.plotly_chart(fig, use_container_width=True)


    st.subheader('Time Series Plot')
    ts_pols = st.multiselect('Select Pollutants', options=df.columns)
    wf_params = {'activate':True,
                 'ws_col':'ws',
                 'wd_col': 'wd',
                 'color': 'black'}
    
    group_data = st.checkbox('Group Data')
    stack_data = st.checkbox('Stack Data')
    normalise_data = st.checkbox('Normalise Data')
    t_av = st.text_input('Select Time Average Strategy (leave blank for none)', value=None, key = 'tp_tav')
    #st.write(f'T:{t_av}')
    #st.write(len(t_av))
    #if len(t_av == 0):
    #    t_av = None
    if ts_pols:
        t = time_plot(df, columns_to_plot=ts_pols, stack_data=stack_data, group_data=group_data, normalize=normalise_data, averaging_period=t_av)
        st.plotly_chart(t, use_container_width=True)
    else:
        st.info('Please select pollutants for analysis')

    st.subheader('Map of Site')
    m = map_sites(metadata, site)
    st.plotly_chart(m, use_container_width=True)
    
    st.subheader('Polar Cluster')
    pc1, pc2 = st.columns(2)
    with pc1:
        n_clusters = st.number_input('Select number of clusters', min_value=1, max_value=12, value=6)
    with pc2:
        which_pols = st.multiselect('Select Pollutants', options=df.columns, key='polar_pols')
    
    if which_pols:
        pc = polar_cluster(df, n_clusters=n_clusters, feature_cols=which_pols)
        st.plotly_chart(pc, use_container_width=True)
    else:
        st.info('Please select columns for polar cluster plot')
        
    
    st.subheader('Polar Plot')
    which_pol = st.selectbox('Select pollutant', options=df.columns.drop(['date', 'ws', 'wd', 'site', 'code', 'date_time']))
    pp = polar_plot(df, conc_col=which_pol)
    st.plotly_chart(pp, use_container_width=True)

    st.subheader('Polar Frequency Plot')
    pr = polar_frequency_plot(df)
    st.plotly_chart(pr, use_container_width=True)

    st.subheader('Wind Rose')
    wr, wrd = wind_rose(df)
    st.plotly_chart(wr, use_container_width=True)
    with st.expander('Wind Rose Data'):
        st.dataframe(wrd, use_container_width=True)
    
    #colour_dict = {'ts':'red', 'rug':'green', 'histogram':'blue'}
    st.subheader('Summary Plot')
    fig, res = summary_plot(df)

    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander('Summary Data'):
        st.dataframe(res, use_container_width=True)
    
    st.subheader('Calendar Plot')
    cal = calendar(df, 'NO2')
    st.plotly_chart(cal, use_container_width=True)
    

    
if __name__ == "__main__":
    main()

