# --- 1. Library Imports ---
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from pyproj import Transformer
import os
import ast
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- 2. Page Configuration & Premium Styling ---
st.set_page_config(layout="wide", page_title="AI 지하수 오염 분석 시스템", page_icon="💧")

# Premium CSS with Full Theme Awareness (Light/Dark Mode)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Streamlit 테마 변수를 사용하여 색상 동적으로 설정 */
    :root {
        --text-color: var(--text-color);
        --primary-color: var(--primary-color);
        --secondary-bg: var(--secondary-background-color);
        --card-border-color: rgba(128, 128, 128, 0.2);
        --card-shadow: rgba(0, 0, 0, 0.08);
    }
    
    .premium-header {
        background: var(--secondary-bg);
        border: 1px solid var(--card-border-color);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px var(--card-shadow);
        animation: slideInDown 1s ease-out;
    }
    
    .premium-metric {
        background-color: var(--secondary-bg);
        border: 1.5px solid var(--card-border-color);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 32px var(--card-shadow);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    .premium-metric:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border-color: var(--primary-color);
    }
    
    .metric-number {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: countUp 1.5s ease-out;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 500;
    }
    
    .insight-card {
        background: var(--secondary-bg);
        border: 1px solid var(--card-border-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px var(--card-shadow);
    }
    
    .premium-sidebar {
        background: var(--secondary-bg);
        border: 1px solid var(--card-border-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        animation: fadeInLeft 0.8s ease-out;
    }
    .premium-sidebar h3 {
        color: var(--text-color) !important;
        text-align: center;
        font-weight: 600;
    }

    .result-banner {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        animation: slideInFromRight 0.6s ease-out;
    }

    .comparison-card {
        background: var(--secondary-bg);
        border: 2px solid var(--card-border-color);
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem;
        box-shadow: 0 8px 32px var(--card-shadow);
        transition: all 0.3s ease;
    }
    
    .comparison-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px var(--card-shadow);
    }
    
    .region-tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
    
    .region-a {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    
    .region-b {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
    }

    /* Animations */
    @keyframes slideInDown { from { opacity: 0; transform: translateY(-50px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes countUp { from { opacity: 0; transform: scale(0.5); } to { opacity: 1; transform: scale(1); } }
    @keyframes fadeInLeft { from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes slideInFromRight { from { opacity: 0; transform: translateX(50px); } to { opacity: 1; transform: translateX(0); } }
</style>
""", unsafe_allow_html=True)

# Matplotlib 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    st.sidebar.warning("⚠️ Malgun Gothic 폰트를 찾을 수 없습니다.")

# --- 3. Enhanced Data Loading ---
@st.cache_data
def load_analysis_results(project_root):
    """Premium data loading with comprehensive error handling"""
    data_folder = os.path.join(project_root, 'data')
    report_path = os.path.join(data_folder, 'individual_source_type_report.csv')
    geojson_path = os.path.join(data_folder, 'all_probable_source_areas.geojson')
    ew_path = os.path.join(data_folder, 'early_warning_results.csv')

    try:
        df_report = pd.read_csv(report_path, encoding='utf-8-sig')
        probable_source_area_gdf = gpd.read_file(geojson_path)
        ew_df = pd.read_csv(ew_path, encoding='utf-8-sig')
        ew_df_filtered = ew_df[ew_df['초과항목수'] > 0].copy()
        
    except FileNotFoundError as e:
        st.error(f"🚨 필수 결과 파일을 찾을 수 없습니다: {e.filename}")
        st.info("💡 Jupyter Notebook에서 모든 분석을 실행하여 결과 파일을 먼저 생성해주세요.")
        st.stop()
        
    return df_report, probable_source_area_gdf, ew_df_filtered

# --- 4. Premium Visualization Functions ---
def create_premium_metrics(filtered_report, filtered_ew):
    """Create premium animated metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    analysis_success_rate = (len(filtered_report) / max(len(filtered_ew), 1)) * 100

    metrics = [
        ("🏭", len(filtered_report), "분석된 오염원"),
        ("⚠️", len(filtered_ew), "오염 감지 지점"),
        ("📊", len(filtered_report['추정 원인'].unique()) if not filtered_report.empty else 0, "오염원 유형"),
        ("📈", f"{analysis_success_rate:.1f}%", "분석 성공률")
    ]

    for i, (icon, value, label) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="premium-metric">
                <div class="metric-icon" style="font-size: 2.5rem; margin-bottom: 1rem;">{icon}</div>
                <div class="metric-number">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def create_premium_charts(filtered_report):
    """Create premium interactive charts with theme awareness"""
    if filtered_report.empty:
        st.markdown("""
        <div class="insight-card">
            <h4>📊 데이터 부족</h4>
            <p>선택된 지역에 대한 분석 결과가 없습니다. 다른 지역을 선택해보세요.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    cause_counts = filtered_report['추정 원인'].value_counts()

    col1, col2 = st.columns([1, 1])

    with col1:
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
        fig_donut = go.Figure(data=[go.Pie(
            labels=cause_counts.index, values=cause_counts.values, hole=0.5,
            hovertemplate="<b>%{label}</b><br>감지 건수: %{value}<br>비율: %{percent}<extra></extra>",
            textinfo="label+percent", textposition="outside",
            marker=dict(colors=colors, line=dict(color='var(--background-color)', width=3))
        )])
        fig_donut.update_layout(
            title={'text': "🎯 오염원 유형별 분포 현황", 'x': 0.5, 'font': {'size': 18}},
            height=450, showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color= "var(--text-color)",
            margin=dict(t=80, b=20, l=20, r=20),
            annotations=[dict(text=f'<b>총 {len(filtered_report)}건</b>', x=0.5, y=0.5, font_size=16, showarrow=False, font_color='var(--primary-color)')]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        fig_bar = go.Figure(data=[go.Bar(
            x=cause_counts.values, y=cause_counts.index, orientation='h',
            marker=dict(color=cause_counts.values, colorscale='Viridis'),
            text=cause_counts.values, textposition='outside'
        )])
        fig_bar.update_layout(
            title={'text': "📈 오염원별 감지 빈도", 'x': 0.5, 'font': {'size': 18}},
            height=450, xaxis_title="감지 건수 (건)", yaxis_title="오염원 유형",
            showlegend=False, margin=dict(t=80, b=60, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color= "var(--text-color)"
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_bar.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- 5. NEW: Regional Comparison Functions ---
def create_comparison_metrics(region_a_data, region_b_data, region_a_name, region_b_name):
    """Create side-by-side comparison metrics"""
    
    # Calculate metrics for both regions
    metrics_a = {
        'sources': len(region_a_data['report']),
        'detections': len(region_a_data['ew']),
        'types': len(region_a_data['report']['추정 원인'].unique()) if not region_a_data['report'].empty else 0,
        'success_rate': (len(region_a_data['report']) / max(len(region_a_data['ew']), 1)) * 100
    }
    
    metrics_b = {
        'sources': len(region_b_data['report']),
        'detections': len(region_b_data['ew']),
        'types': len(region_b_data['report']['추정 원인'].unique()) if not region_b_data['report'].empty else 0,
        'success_rate': (len(region_b_data['report']) / max(len(region_b_data['ew']), 1)) * 100
    }
    
    st.markdown("### 🔄 지역별 핵심 지표 비교")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="comparison-card">
            <div class="region-tag region-a">📍 {region_a_name}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{metrics_a['sources']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">분석된 오염원</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #764ba2;">{metrics_a['detections']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">오염 감지 지점</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #667eea;">{metrics_a['types']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">오염원 유형</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #764ba2;">{metrics_a['success_rate']:.1f}%</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">분석 성공률</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="comparison-card">
            <div class="region-tag region-b">📍 {region_b_name}</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #f093fb;">{metrics_b['sources']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">분석된 오염원</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #f5576c;">{metrics_b['detections']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">오염 감지 지점</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #f093fb;">{metrics_b['types']}</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">오염원 유형</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2rem; font-weight: bold; color: #f5576c;">{metrics_b['success_rate']:.1f}%</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">분석 성공률</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_comparison_charts(region_a_data, region_b_data, region_a_name, region_b_name):
    """Create side-by-side comparison charts"""
    
    if region_a_data['report'].empty and region_b_data['report'].empty:
        st.warning("선택된 두 지역 모두 분석 결과가 없습니다.")
        return
    
    st.markdown("### 📊 오염원 유형 분포 비교")
    
    col1, col2 = st.columns(2)
    
    # Region A Chart
    with col1:
        if not region_a_data['report'].empty:
            cause_counts_a = region_a_data['report']['추정 원인'].value_counts()
            
            fig_a = go.Figure(data=[go.Pie(
                labels=cause_counts_a.index, 
                values=cause_counts_a.values, 
                hole=0.5,
                marker=dict(colors=['#667eea', '#764ba2', '#4facfe', '#43e97b', '#f093fb']),
                hovertemplate="<b>%{label}</b><br>감지 건수: %{value}<br>비율: %{percent}<extra></extra>",
                textinfo="label+percent"
            )])
            
            fig_a.update_layout(
                title={'text': f"📍 {region_a_name}", 'x': 0.5, 'font': {'size': 16}},
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="var(--text-color)",
                annotations=[dict(
                    text=f'<b>총 {len(region_a_data["report"])}건</b>', 
                    x=0.5, y=0.5, 
                    font_size=14, 
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info(f"📍 {region_a_name}: 분석 결과 없음")
    
    # Region B Chart
    with col2:
        if not region_b_data['report'].empty:
            cause_counts_b = region_b_data['report']['추정 원인'].value_counts()
            
            fig_b = go.Figure(data=[go.Pie(
                labels=cause_counts_b.index, 
                values=cause_counts_b.values, 
                hole=0.5,
                marker=dict(colors=['#f093fb', '#f5576c', '#ff6b6b', '#feca57', '#48dbfb']),
                hovertemplate="<b>%{label}</b><br>감지 건수: %{value}<br>비율: %{percent}<extra></extra>",
                textinfo="label+percent"
            )])
            
            fig_b.update_layout(
                title={'text': f"📍 {region_b_name}", 'x': 0.5, 'font': {'size': 16}},
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="var(--text-color)",
                annotations=[dict(
                    text=f'<b>총 {len(region_b_data["report"])}건</b>', 
                    x=0.5, y=0.5, 
                    font_size=14, 
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info(f"📍 {region_b_name}: 분석 결과 없음")
    
    # Combined comparison bar chart
    if not region_a_data['report'].empty or not region_b_data['report'].empty:
        st.markdown("### 📈 지역별 오염원 유형 직접 비교")
        
        # Get all unique causes from both regions
        all_causes = set()
        if not region_a_data['report'].empty:
            all_causes.update(region_a_data['report']['추정 원인'].unique())
        if not region_b_data['report'].empty:
            all_causes.update(region_b_data['report']['추정 원인'].unique())
        
        # Create comparison data
        comparison_data = []
        for cause in all_causes:
            count_a = len(region_a_data['report'][region_a_data['report']['추정 원인'] == cause]) if not region_a_data['report'].empty else 0
            count_b = len(region_b_data['report'][region_b_data['report']['추정 원인'] == cause]) if not region_b_data['report'].empty else 0
            comparison_data.append({
                '오염원': cause,
                region_a_name: count_a,
                region_b_name: count_b
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name=region_a_name,
            x=comparison_df['오염원'],
            y=comparison_df[region_a_name],
            marker_color='#667eea',
            text=comparison_df[region_a_name],
            textposition='outside'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name=region_b_name,
            x=comparison_df['오염원'],
            y=comparison_df[region_b_name],
            marker_color='#f093fb',
            text=comparison_df[region_b_name],
            textposition='outside'
        ))
        
        fig_comparison.update_layout(
            title={'text': f"🔄 {region_a_name} vs {region_b_name} 오염원 유형별 비교", 'x': 0.5},
            xaxis_title="오염원 유형",
            yaxis_title="감지 건수",
            barmode='group',
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="var(--text-color)"
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)

def create_comparison_map(region_a_data, region_b_data, region_a_name, region_b_name):
    """Create comparison map with both regions"""
    st.markdown("### 🗺️ 지역 비교 인터랙티브 맵")
    
    coord_transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    
    # Calculate map center
    all_lats, all_lons = [], []
    
    for region_data in [region_a_data, region_b_data]:
        if not region_data['report'].empty:
            all_lats.extend(region_data['report']['오염원_위도'].tolist())
            all_lons.extend(region_data['report']['오염원_경도'].tolist())
    
    if all_lats and all_lons:
        map_center = [np.mean(all_lats), np.mean(all_lons)]
        zoom_level = 9
    else:
        map_center = [36.5, 127.8]
        zoom_level = 7
    
    m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron")
    
    # Feature groups for each region
    fg_wells_a = folium.FeatureGroup(name=f"🚨 {region_a_name} 오염 감지", show=True).add_to(m)
    fg_sources_a = folium.FeatureGroup(name=f"🤖 {region_a_name} AI 추론", show=True).add_to(m)
    fg_wells_b = folium.FeatureGroup(name=f"🚨 {region_b_name} 오염 감지", show=True).add_to(m)
    fg_sources_b = folium.FeatureGroup(name=f"🤖 {region_b_name} AI 추론", show=True).add_to(m)
    
    # Add Region A markers
    for _, row in region_a_data['ew'].iterrows():
        try:
            lon, lat = coord_transformer.transform(row["TM_X_5186"], row["TM_Y_5186"])
            ex_list = ast.literal_eval(row['초과항목목록']) if pd.notna(row['초과항목목록']) else []
            ex_str = ', '.join(ex_list) if ex_list else "정보 없음"
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; max-width: 300px;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; margin: -0.5rem -0.5rem 1rem -0.5rem; color: white; border-radius: 8px 8px 0 0;">
                    <h4 style="margin: 0; font-size: 1.1rem;">🚨 {region_a_name} 오염 감지</h4>
                </div>
                <div style="padding: 0 0.5rem;">
                    <p style="margin: 0.5rem 0;"><b>📍 관측소:</b> {row['공번호']}</p>
                    <p style="margin: 0.5rem 0;"><b>⚠️ 초과:</b> {row['초과항목수']}개 항목</p>
                    <p style="margin: 0.5rem 0;"><b>🧪 항목:</b> {ex_str}</p>
                </div>
            </div>
            """
            
            folium.CircleMarker([lat, lon], radius=8, color='#667eea', fill=True, fill_color='#667eea', fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=350)
            ).add_to(fg_wells_a)
        except: continue
    
    # Add Region B markers
    for _, row in region_b_data['ew'].iterrows():
        try:
            lon, lat = coord_transformer.transform(row["TM_X_5186"], row["TM_Y_5186"])
            ex_list = ast.literal_eval(row['초과항목목록']) if pd.notna(row['초과항목목록']) else []
            ex_str = ', '.join(ex_list) if ex_list else "정보 없음"
            
            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; max-width: 300px;">
                <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 1rem; margin: -0.5rem -0.5rem 1rem -0.5rem; color: white; border-radius: 8px 8px 0 0;">
                    <h4 style="margin: 0; font-size: 1.1rem;">🚨 {region_b_name} 오염 감지</h4>
                </div>
                <div style="padding: 0 0.5rem;">
                    <p style="margin: 0.5rem 0;"><b>📍 관측소:</b> {row['공번호']}</p>
                    <p style="margin: 0.5rem 0;"><b>⚠️ 초과:</b> {row['초과항목수']}개 항목</p>
                    <p style="margin: 0.5rem 0;"><b>🧪 항목:</b> {ex_str}</p>
                </div>
            </div>
            """
            
            folium.CircleMarker([lat, lon], radius=8, color='#f093fb', fill=True, fill_color='#f093fb', fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=350)
            ).add_to(fg_wells_b)
        except: continue
    
    # Add Region A source markers
    for _, row in region_a_data['report'].iterrows():
        popup_html = f"""
        <div style="font-family: 'Inter', sans-serif; max-width: 320px;">
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; margin: -0.5rem -0.5rem 1rem -0.5rem; color: white; border-radius: 8px 8px 0 0;">
                <h4 style="margin: 0; font-size: 1.1rem;">🤖 {region_a_name} AI 분석</h4>
            </div>
            <div style="padding: 0 0.5rem;">
                <p style="margin: 0.5rem 0;"><b>🎯 관측소:</b> {row['오염 관측소']}</p>
                <p style="margin: 0.5rem 0;"><b>🔬 추정 오염원:</b> {row['추정 원인']}</p>
                <p style="margin: 0.5rem 0;"><b>🎯 정확도:</b> {row['확률']}</p>
            </div>
        </div>
        """
        folium.Marker([row['오염원_위도'], row['오염원_경도']], 
            popup=folium.Popup(popup_html, max_width=370),
            icon=folium.Icon(color='blue', icon='brain', prefix='fa')
        ).add_to(fg_sources_a)
    
    # Add Region B source markers
    for _, row in region_b_data['report'].iterrows():
        popup_html = f"""
        <div style="font-family: 'Inter', sans-serif; max-width: 320px;">
            <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 1rem; margin: -0.5rem -0.5rem 1rem -0.5rem; color: white; border-radius: 8px 8px 0 0;">
                <h4 style="margin: 0; font-size: 1.1rem;">🤖 {region_b_name} AI 분석</h4>
            </div>
            <div style="padding: 0 0.5rem;">
                <p style="margin: 0.5rem 0;"><b>🎯 관측소:</b> {row['오염 관측소']}</p>
                <p style="margin: 0.5rem 0;"><b>🔬 추정 오염원:</b> {row['추정 원인']}</p>
                <p style="margin: 0.5rem 0;"><b>🎯 정확도:</b> {row['확률']}</p>
            </div>
        </div>
        """
        folium.Marker([row['오염원_위도'], row['오염원_경도']], 
            popup=folium.Popup(popup_html, max_width=370),
            icon=folium.Icon(color='pink', icon='brain', prefix='fa')
        ).add_to(fg_sources_b)
    
    folium.LayerControl().add_to(m)
    st_folium(m, width='100%', height=650)

# --- 6. Main Premium App ---
def main():
    # 헤더 스타일 옵션 - 원하는 것으로 선택하세요!
    
    # 옵션 1: 간결한 한 줄 (추천)
    st.markdown("""
    <div class="premium-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">💧 AI 지하수 오염원 분석 시스템</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # 옵션 2: 한 줄 + 아이콘 강조
    # st.markdown("""
    # <div class="premium-header">
    #     <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">
    #         💧🤖 AI 지하수 오염원 분석 시스템
    #     </h1>
    # </div>
    # """, unsafe_allow_html=True)
    
    # 옵션 3: 더 짧고 임팩트 있게
    # st.markdown("""
    # <div class="premium-header">
    #     <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">💧 지하수 AI 분석</h1>
    # </div>
    # """, unsafe_allow_html=True)
    
    # 옵션 4: 기존 두 줄 유지 (현재)
    # st.markdown("""
    # <div class="premium-header">
    #     <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">💧 AI 기반 지하수 오염원 역추적 시스템</h1>
    #     <p style="font-size: 1.3rem; margin: 1rem 0 0 0; opacity: 0.8; font-weight: 400;">
    #         🤖 차세대 딥러닝 모델을 활용한 지능형 오염원 분석 플랫폼
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)

    with st.spinner('🔄 AI 분석 데이터를 로딩하는 중...'):
        project_root = os.getcwd()
        df_report, probable_source_area_gdf, ew_df_filtered = load_analysis_results(project_root)

    # Sidebar configuration
    st.sidebar.markdown("""
    <div class="premium-sidebar">
        <h3>⚙️ 분석 옵션 설정</h3>
    </div>
    """, unsafe_allow_html=True)

    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "**🔍 분석 모드 선택**",
        ["🏘️ 단일 지역 분석", "🔄 지역 비교 분석"],
        help="단일 지역을 자세히 분석하거나 두 지역을 비교할 수 있습니다."
    )

    sigungu_list = ["전체"] + sorted(ew_df_filtered['시군구명'].dropna().unique().tolist())

    if analysis_mode == "🏘️ 단일 지역 분석":
        # 기존 단일 지역 분석 로직
        selected_sigungu = st.sidebar.selectbox(
            "**🎯 분석 대상 지역 선택**",
            sigungu_list,
            help="상세 분석 결과를 확인할 지역을 선택하세요"
        )

        if selected_sigungu == "전체":
            filtered_report = df_report
            filtered_ew = ew_df_filtered
        else:
            filtered_report = df_report[df_report['소재지'] == selected_sigungu]
            filtered_ew = ew_df_filtered[ew_df_filtered['시군구명'] == selected_sigungu]

        st.sidebar.markdown(f"""
        <div class="premium-sidebar">
            <h4 style="color: var(--primary-color); margin: 0 0 1rem 0; text-align:center;">📍 {selected_sigungu} 현황</h4>
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 8px; color: white; text-align: center;">
                <p style="margin: 0;">🚨 <strong>{len(filtered_ew)}건</strong> 오염 감지</p>
                <p style="margin: 0.5rem 0 0 0;">🔍 <strong>{len(filtered_report)}개소</strong> 분석 완료</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main title based on selection
        if selected_sigungu == "전체":
            st.markdown("## 🗺️ 지능형 지하수 오염 통합 관제 대시보드")
        else:
            st.markdown(f"## 🗺️ {selected_sigungu} | 오염 현황 상세 분석")

        # Existing tabs
        tab_report, tab_map, tab_data = st.tabs([
            "📊 **종합 리포트**", 
            "🗺️ **인터랙티브 맵**", 
            "📄 **상세 데이터 조회**"
        ])

        with tab_report:
            st.markdown("### 🎯 핵심 지표 요약")
            create_premium_metrics(filtered_report, filtered_ew)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 📊 오염원 유형 통계")
            create_premium_charts(filtered_report)

        with tab_map:
            st.markdown("### 🗺️ AI 분석 결과 지도")
            coord_transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
            
            if filtered_report.empty and selected_sigungu != "전체":
                map_center, zoom_level = [36.5, 127.8], 7
                st.warning("선택하신 지역에는 분석된 오염원이 없습니다.")
            else:
                map_center = [filtered_report['오염원_위도'].mean(), filtered_report['오염원_경도'].mean()] if not filtered_report.empty else [36.5, 127.8]
                zoom_level = 11 if selected_sigungu != "전체" else 7

            m = folium.Map(location=map_center, zoom_start=zoom_level, tiles="CartoDB positron")
            
            fg_wells = folium.FeatureGroup(name="🚨 오염 감지 관측소", show=True).add_to(m)
            fg_sources = folium.FeatureGroup(name="🤖 AI 추론 오염원", show=True).add_to(m)
            fg_area = folium.FeatureGroup(name="🎯 종합 위험 영역", show=(selected_sigungu == "전체")).add_to(m)
            
            # Enhanced well markers
            for _, row in filtered_ew.iterrows():
                try:
                    lon, lat = coord_transformer.transform(row["TM_X_5186"], row["TM_Y_5186"])
                    ex_list = ast.literal_eval(row['초과항목목록']) if pd.notna(row['초과항목목록']) else []
                    ex_str = ', '.join(ex_list) if ex_list else "정보 없음"
                    
                    popup_html = f"""
                    <div style="font-family: 'Inter', sans-serif; max-width: 300px;">
                        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; margin: -0.5rem -0.5rem 1rem -0.5rem; color: white; border-radius: 8px 8px 0 0;">
                            <h4 style="margin: 0; font-size: 1.1rem;">🚨 오염 감지 정보</h4>
                        </div>
                        <div style="padding: 0 0.5rem;">
                            <p style="margin: 0.5rem 0;"><b>📍 관측소 ID:</b> {row['공번호']}</p>
                            <p style="margin: 0.5rem 0;"><b>🏘️ 위치:</b> {row['시군구명']}</p>
                            <p style="margin: 0.5rem 0;"><b>📅 검사일:</b> {row['조사시기']}</p>
                            <p style="margin: 0.5rem 0;"><b>⚠️ 기준 초과:</b> <span style="color: #e74c3c; font-weight: bold;">{row['초과항목수']}개 항목</span></p>
                            <p style="margin: 0.5rem 0;"><b>🧪 상세 항목:</b> {ex_str}</p>
                        </div>
                    </div>
                    """
                    
                    color = 'red' if row['초과항목수'] >= 3 else 'orange' if row['초과항목수'] == 2 else 'blue'
                    size = 10 if row['초과항목수'] >= 3 else 8 if row['초과항목수'] == 2 else 6
                    
                    folium.CircleMarker([lat, lon], radius=size, color=color, fill=True, fill_color=color, fill_opacity=0.8,
                        popup=folium.Popup(popup_html, max_width=350), tooltip=f"🚨 {row['공번호']} ({row['초과항목수']}개 기준 초과)"
                    ).add_to(fg_wells)
                except: continue
            
            # Enhanced source markers
            for _, row in filtered_report.iterrows():
                popup_html = f"""
                <div style="font-family: 'Inter', sans-serif; max-width: 320px;">
                    <div style="background: linear-gradient(135deg, #8e44ad, #3498db); padding: 1rem; margin: -0.5rem -0.5rem 1rem -0.5rem; color: white; border-radius: 8px 8px 0 0;">
                        <h4 style="margin: 0; font-size: 1.1rem;">🤖 AI 분석 결과</h4>
                    </div>
                    <div style="padding: 0 0.5rem;">
                        <p style="margin: 0.5rem 0;"><b>🎯 연결 관측소:</b> {row['오염 관측소']}</p>
                        <p style="margin: 0.5rem 0;"><b>📍 분석 지역:</b> {row['소재지']}</p>
                        <p style="margin: 0.5rem 0;"><b>🔬 추정 오염원:</b> <span style="color: #e74c3c; font-weight: bold;">{row['추정 원인']}</span></p>
                        <p style="margin: 0.5rem 0;"><b>🎯 분석 정확도:</b> <span style="color: #27ae60; font-weight: bold;">{row['확률']}</span></p>
                    </div>
                </div>
                """
                folium.Marker([row['오염원_위도'], row['오염원_경도']], popup=folium.Popup(popup_html, max_width=370),
                    tooltip=f"🤖 AI 분석: {row['추정 원인']} ({row['확률']})", icon=folium.Icon(color='purple', icon='brain', prefix='fa')
                ).add_to(fg_sources)
            
            # Enhanced area polygon
            if selected_sigungu == "전체" and not probable_source_area_gdf.empty:
                area_km2 = probable_source_area_gdf.geometry.area.iloc[0] / 1e6
                top3_causes_str = df_report['추정 원인'].value_counts().nlargest(3).to_string().replace('\n', '<br>')
                popup_html = f"""
                <div style="font-family: 'Inter', sans-serif; max-width: 350px;">
                     <div style="background: linear-gradient(135deg, #d35400, #e74c3c); padding: 1rem; margin: -0.5rem -0.5rem 1rem -0.5rem; color: white; border-radius: 8px 8px 0 0;"><h4 style="margin: 0; font-size: 1.1rem;">🎯 종합 위험 영역</h4></div>
                     <div style="padding: 0 0.5rem;">
                        <p style="margin: 0.5rem 0;"><b>📐 총 면적:</b> {area_km2:.2f} km²</p>
                        <p style="margin: 0.5rem 0;"><b>🏭 분석 완료:</b> {len(df_report)}개소</p>
                        <p style="margin: 0.5rem 0;"><b>📊 주요 오염원 Top 3:</b></p>
                        <div style="background: #f8f9fa; color: #333; padding: 0.8rem; border-radius: 5px; font-family: monospace; font-size: 0.9rem;">{top3_causes_str}</div>
                    </div>
                </div>
                """
                folium.GeoJson(probable_source_area_gdf.to_crs("EPSG:4326"),
                    style_function=lambda x: {'color': '#e74c3c', 'weight': 3, 'fillOpacity': 0.15, 'dashArray': '10, 5'},
                    tooltip='🎯 종합 위험 영역 (클릭하여 상세정보 확인)', popup=folium.Popup(popup_html, max_width=400)).add_to(fg_area)

            folium.LayerControl().add_to(m)
            st_folium(m, width='100%', height=650)

        with tab_data:
            st.markdown("### 📄 상세 데이터 조회 및 필터링")
            if not filtered_report.empty:
                col1, col2 = st.columns(2)
                with col1:
                    cause_filter = st.multiselect(
                        "**🏭 오염원 유형 선택**", options=sorted(filtered_report['추정 원인'].unique()),
                        default=sorted(filtered_report['추정 원인'].unique())
                    )
                with col2:
                    try:
                        prob_values = filtered_report['확률'].str.rstrip('%').astype(float) if filtered_report['확률'].dtype == 'object' else filtered_report['확률'].astype(float)
                        min_prob, max_prob = int(prob_values.min()), int(prob_values.max())
                        prob_threshold = st.slider("**🎯 최소 분석 정확도 (%)**", min_prob, max_prob, min_prob, 5)
                    except: prob_threshold = 0
                
                display_data = filtered_report
                if cause_filter:
                    display_data = display_data[display_data['추정 원인'].isin(cause_filter)]
                
                try:
                    if 'prob_values' in locals():
                        current_probs = display_data['확률'].str.rstrip('%').astype(float) if display_data['확률'].dtype == 'object' else display_data['확률'].astype(float)
                        display_data = display_data[current_probs >= prob_threshold]
                except Exception:
                    display_data = pd.DataFrame()

                st.markdown(f"""
                <div class="result-banner">
                    <h3 style="margin: 0;">📊 필터링 결과: 총 {len(display_data)}건</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if not display_data.empty:
                    st.dataframe(display_data, use_container_width=True, height=500)
                    csv = display_data.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📊 CSV 파일로 다운로드", csv, f"AI_분석결과.csv", "text/csv")
                else:
                    st.warning("선택한 조건에 맞는 데이터가 없습니다.")
            else:
                st.info("선택된 지역에 대한 분석 데이터가 없습니다.")

    else:  # 지역 비교 분석 모드
        st.markdown("## 🔄 지역별 오염 양상 비교 분석")
        
        # Region selection for comparison
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.markdown("**📍 지역 A 선택**")
            region_a = st.selectbox(
                "첫 번째 비교 지역",
                [region for region in sigungu_list if region != "전체"],
                key="region_a"
            )
        
        with col2:
            st.markdown("**📍 지역 B 선택**")
            available_regions_b = [region for region in sigungu_list if region != "전체" and region != region_a]
            region_b = st.selectbox(
                "두 번째 비교 지역",
                available_regions_b,
                key="region_b"
            )

        # Prepare data for both regions
        region_a_report = df_report[df_report['소재지'] == region_a]
        region_a_ew = ew_df_filtered[ew_df_filtered['시군구명'] == region_a]
        region_a_data = {'report': region_a_report, 'ew': region_a_ew}

        region_b_report = df_report[df_report['소재지'] == region_b]
        region_b_ew = ew_df_filtered[ew_df_filtered['시군구명'] == region_b]
        region_b_data = {'report': region_b_report, 'ew': region_b_ew}

        # Sidebar summary
        st.sidebar.markdown(f"""
        <div class="premium-sidebar">
            <h4 style="color: var(--primary-color); margin: 0 0 1rem 0; text-align:center;">🔄 비교 현황</h4>
            <div style="margin-bottom: 1rem;">
                <div class="region-tag region-a">{region_a}</div>
                <div style="background: #667eea; padding: 0.5rem; border-radius: 8px; color: white; margin-top: 0.5rem; font-size: 0.9rem;">
                    🚨 {len(region_a_ew)}건 감지 | 🔍 {len(region_a_report)}개소 분석
                </div>
            </div>
            <div>
                <div class="region-tag region-b">{region_b}</div>
                <div style="background: #f093fb; padding: 0.5rem; border-radius: 8px; color: white; margin-top: 0.5rem; font-size: 0.9rem;">
                    🚨 {len(region_b_ew)}건 감지 | 🔍 {len(region_b_report)}개소 분석
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison tabs
        tab_comparison, tab_map_comparison, tab_insights = st.tabs([
            "📊 **비교 리포트**", 
            "🗺️ **비교 지도**", 
            "💡 **인사이트**"
        ])

        with tab_comparison:
            create_comparison_metrics(region_a_data, region_b_data, region_a, region_b)
            st.markdown("<hr>", unsafe_allow_html=True)
            create_comparison_charts(region_a_data, region_b_data, region_a, region_b)

        with tab_map_comparison:
            create_comparison_map(region_a_data, region_b_data, region_a, region_b)

        with tab_insights:
            st.markdown("### 💡 AI 기반 지역 비교 인사이트")
            
            # Generate insights
            insights = []
            
            # 감지 건수 비교
            if len(region_a_ew) > len(region_b_ew):
                insights.append(f"🚨 **{region_a}**가 **{region_b}**보다 {len(region_a_ew) - len(region_b_ew)}건 더 많은 오염이 감지되었습니다.")
            elif len(region_b_ew) > len(region_a_ew):
                insights.append(f"🚨 **{region_b}**가 **{region_a}**보다 {len(region_b_ew) - len(region_a_ew)}건 더 많은 오염이 감지되었습니다.")
            else:
                insights.append(f"⚖️ **{region_a}**와 **{region_b}** 모두 동일한 {len(region_a_ew)}건의 오염이 감지되었습니다.")
            
            # 오염원 유형 다양성 비교
            types_a = len(region_a_data['report']['추정 원인'].unique()) if not region_a_data['report'].empty else 0
            types_b = len(region_b_data['report']['추정 원인'].unique()) if not region_b_data['report'].empty else 0
            
            if types_a > types_b:
                insights.append(f"🔬 **{region_a}**에서 더 다양한 오염원 유형({types_a}종)이 발견되어 복합적인 오염 양상을 보입니다.")
            elif types_b > types_a:
                insights.append(f"🔬 **{region_b}**에서 더 다양한 오염원 유형({types_b}종)이 발견되어 복합적인 오염 양상을 보입니다.")
            
            # 주요 오염원 비교
            if not region_a_data['report'].empty and not region_b_data['report'].empty:
                top_cause_a = region_a_data['report']['추정 원인'].value_counts().index[0]
                top_cause_b = region_b_data['report']['추정 원인'].value_counts().index[0]
                
                if top_cause_a == top_cause_b:
                    insights.append(f"⭐ 두 지역 모두 **{top_cause_a}**이 주요 오염원으로 나타났습니다.")
                else:
                    insights.append(f"🎯 **{region_a}**의 주요 오염원은 **{top_cause_a}**, **{region_b}**의 주요 오염원은 **{top_cause_b}**로 서로 다른 양상을 보입니다.")
            
            # 분석 성공률 비교
            success_rate_a = (len(region_a_data['report']) / max(len(region_a_data['ew']), 1)) * 100
            success_rate_b = (len(region_b_data['report']) / max(len(region_b_data['ew']), 1)) * 100
            
            if abs(success_rate_a - success_rate_b) > 10:
                if success_rate_a > success_rate_b:
                    insights.append(f"📈 **{region_a}**의 AI 분석 성공률({success_rate_a:.1f}%)이 **{region_b}**({success_rate_b:.1f}%)보다 높아 더 정확한 오염원 추적이 가능합니다.")
                else:
                    insights.append(f"📈 **{region_b}**의 AI 분석 성공률({success_rate_b:.1f}%)이 **{region_a}**({success_rate_a:.1f}%)보다 높아 더 정확한 오염원 추적이 가능합니다.")
            
            # Display insights
            for i, insight in enumerate(insights, 1):
                st.markdown(f"""
                <div class="insight-card">
                    <h4 style="color: var(--primary-color); margin-bottom: 1rem;">💡 인사이트 {i}</h4>
                    <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{insight}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 권장사항
            st.markdown("### 🎯 관리 권장사항")
            
            recommendations = []
            
            if len(region_a_ew) > len(region_b_ew) * 1.5:
                recommendations.append(f"**{region_a}** 지역에 대한 집중적인 모니터링과 오염원 차단 조치가 필요합니다.")
            elif len(region_b_ew) > len(region_a_ew) * 1.5:
                recommendations.append(f"**{region_b}** 지역에 대한 집중적인 모니터링과 오염원 차단 조치가 필요합니다.")
            
            if types_a > types_b:
                recommendations.append(f"**{region_a}** 지역은 다양한 오염원에 대한 종합적인 대응 전략이 필요합니다.")
            elif types_b > types_a:
                recommendations.append(f"**{region_b}** 지역은 다양한 오염원에 대한 종합적인 대응 전략이 필요합니다.")
            
            if success_rate_a < 70:
                recommendations.append(f"**{region_a}** 지역의 AI 분석 정확도 향상을 위한 추가 데이터 수집이 필요합니다.")
            if success_rate_b < 70:
                recommendations.append(f"**{region_b}** 지역의 AI 분석 정확도 향상을 위한 추가 데이터 수집이 필요합니다.")
            
            recommendations.append("두 지역 간 오염 패턴 차이를 활용하여 맞춤형 관리 전략을 수립하는 것이 효과적입니다.")
            recommendations.append("정기적인 비교 분석을 통해 지역별 오염 추세 변화를 모니터링해야 합니다.")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="insight-card" style="border-left: 4px solid var(--primary-color);">
                    <h4 style="color: var(--primary-color); margin-bottom: 1rem;">🎯 권장사항 {i}</h4>
                    <p style="margin: 0; font-size: 1.1rem; line-height: 1.6;">{rec}</p>
                </div>
                """, unsafe_allow_html=True)

    # 공통 sidebar 정보
    st.sidebar.info("💎 테마 변경: 메뉴 > Settings > Theme")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: var(--secondary-bg); border-radius: 8px; margin-top: 1rem;">
        <h4 style="margin: 0; color: var(--primary-color);">🤖 AI 분석 현황</h4>
        <p style="margin: 0.5rem 0; font-size: 0.9rem;">
            총 <strong>{}</strong>개 지역에서<br>
            <strong>{}</strong>건의 오염원을 분석했습니다.
        </p>
    </div>
    """.format(
        len(ew_df_filtered['시군구명'].unique()),
        len(df_report)
    ), unsafe_allow_html=True)

# --- 7. Main Guard ---
if __name__ == "__main__":
    main()