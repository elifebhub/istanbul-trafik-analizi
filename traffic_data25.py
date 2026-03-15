import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import holidays
import gc # RAM'i temizlemek

# 1. AYARLAR VE TATİL TAKVİMİ
st.set_page_config(page_title="Trafik Analizi", layout="wide", page_icon="🚦")
tr_holidays = holidays.Turkey()

ILCE_KOORDINAT = {
    # --- Avrupa Yakası ---
    "Beşiktaş": (41.0422, 29.0082),
    "Şişli (Mecidiyeköy)": (41.0600, 28.9870),
    "Fatih (Eminönü)": (41.0122, 28.9390),
    "Bakırköy": (40.9782, 28.8710),
    "Beylikdüzü": (41.0035, 28.6410),
    "Esenyurt": (41.0340, 28.6820),
    "Sarıyer": (41.1667, 29.0500),
    "Bağcılar": (41.0344, 28.8333),
    "Küçükçekmece": (41.0011, 28.7711),
    "Kağıthane": (41.0811, 28.9733),
    "Zeytinburnu": (40.9881, 28.9036),
    # --- Anadolu Yakası ---
    "Kadıköy": (40.9910, 29.0270),
    "Üsküdar": (41.0260, 29.0150),
    "Ataşehir": (40.9847, 29.1067),
    "Ümraniye": (41.0256, 29.1161),
    "Maltepe": (40.9246, 29.1311),
    "Kartal": (40.8886, 29.1856),
    "Pendik": (40.8769, 29.2347),
    "Tuzla": (40.8153, 29.3094),
    "Beykoz": (41.1167, 29.1000)
}


# 2. VERİ VE MODEL MOTORU
@st.cache_data
def verileri_hazirla(yuklenen_dosyalar):
    df_listesi = []
    # RAM Tasarrufu: Sadece bize lazım olan 5 sütunu okuyoruz
    sutun_haritasi = {
        'DATE_TIME': 'Tarih_Saat', 'LATITUDE': 'lat', 'LONGITUDE': 'lon',
        'AVERAGE_SPEED': 'Hiz', 'NUMBER_OF_VEHICLES': 'Arac'
    }

    for dosya in yuklenen_dosyalar:
        try:
            temp_df = pd.read_csv(dosya, encoding="utf-8-sig")
        except:
            temp_df = pd.read_csv(dosya, encoding="ISO-8859-9")
        
        # Gereksiz sütunları (Geohash vb.) hemen çöpe atıyoruz
        temp_df = temp_df[list(sutun_haritasi.keys())].rename(columns=sutun_haritasi)
        
        # RAM Tasarrufu: Veri tiplerini float64'ten float32'ye düşür (Yer kazancı %50)
        temp_df['lat'] = temp_df['lat'].astype('float32')
        temp_df['lon'] = temp_df['lon'].astype('float32')
        temp_df['Hiz'] = temp_df['Hiz'].astype('int16')
        
        df_listesi.append(temp_df)

    # Tüm dosyaları birleştir ve listeyi silerek RAM'de yer aç
    df = pd.concat(df_listesi, ignore_index=True)
    del df_listesi 
    gc.collect()

    # Tarih işlemleri
    df['Tarih_Saat'] = pd.to_datetime(df['Tarih_Saat'], errors='coerce')
    df = df.dropna(subset=['Tarih_Saat', 'Hiz', 'Arac'])

    # Zaman özelliklerini int8 yaparak belleği daha da rahatlatıyoruz
    df['Saat'] = df['Tarih_Saat'].dt.hour.astype('int8')
    df['Ay'] = df['Tarih_Saat'].dt.month.astype('int8')
    df['Gün'] = df['Tarih_Saat'].dt.date
    df['Gun_No'] = df['Tarih_Saat'].dt.weekday.astype('int8')
    df['Is_Holiday'] = df['Gün'].apply(lambda x: 1 if x in tr_holidays else 0).astype('int8')

    def renk_ata(h):
        if h < 30: return [255, 0, 0, 160]
        elif h < 60: return [255, 200, 0, 160]
        else: return [0, 255, 0, 160]

    df['Renk'] = df['Hiz'].apply(renk_ata)
    return df


@st.cache_resource
def model_egit(veri):
    # Çoklu dosyadan gelen milyonlarca satır yerine en kaliteli 50.000 örneği seçiyoruz
    ornek = veri.sample(n=min(50000, len(veri)), random_state=42)
    
    # Giriş parametrelerine 'Ay' eklendi (Mevsimsel tahmin için şart!)
    X = ornek[['Saat', 'Gun_No', 'Arac', 'lat', 'lon', 'Is_Holiday', 'Ay']]
    y = ornek['Hiz']
    
    # n_estimators'ı 30 yaparak hızı artırdık
    rf = RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Kullanılan geçici veriyi RAM'den siliyoruz
    del ornek
    gc.collect()
    
    return rf


# 3. YAN MENÜ
st.sidebar.title("🧭 Kontrol Paneli")
yuklenen_dosyalar = st.sidebar.file_uploader("Analiz için .csv dosyalarını buraya yükleyin", type=["csv"], accept_multiple_files=True)

if yuklenen_dosyalar:
    df = verileri_hazirla(yuklenen_dosyalar)
    st.sidebar.success(f"{len(yuklenen_dosyalar)} dosya birleştirildi.")
    
    # ---  TARİH SEÇİMİ ---
    # Tüm tarihleri değil, sadece mevcut olan benzersiz tarihleri alıyoruz
    mevcut_tarihler = sorted(df['Gün'].unique())
    secilen_gün = st.sidebar.selectbox("📅 Analiz Tarihi:", mevcut_tarihler)
    
    # ---  Veriyi ana bellekte hemen küçült ---
    # Kullanıcı tarihi seçtiği an, RAM'de duran milyonlarca satırı 
    # sadece o güne (yaklaşık 30-40 bin satır) indiriyoruz.
    df_gunluk = df[df['Gün'] == secilen_gün].copy()
    
    # (Bu satır RAM'i %90 rahatlatır)

    secilen_saat = st.sidebar.slider("⏰ Saat Dilimi:", 0, 23, 8)
    secilen_yaka = st.sidebar.selectbox("📍 Bölge:", ("Tümü", "Avrupa Yakası", "Anadolu Yakası"))
    
    # Filtreleme işlemlerini artık milyonlarca satır değil, 
    # sadece seçilen günün (df_gunluk) üzerinden yapıyoruz:
    if secilen_yaka == "Avrupa Yakası": 
        df_bolge = df_gunluk[df_gunluk['lon'] < 29.0]
    elif secilen_yaka == "Anadolu Yakası": 
        df_bolge = df_gunluk[df_gunluk['lon'] >= 29.0]
    else: 
        df_bolge = df_gunluk

    saatlik_veri = df_bolge[df_bolge['Saat'] == secilen_saat].copy()


    def sev_bul(h):
        return "🚨 Yoğun" if h < 30 else ("🟡 Akıcı" if h < 60 else "🟢 Açık")


    saatlik_veri['Durum'] = saatlik_veri['Hiz'].apply(sev_bul)
    saatlik_veri = saatlik_veri[saatlik_veri['Durum'].isin(trafik_filtresi)]

    # 5. SEKMELER
    tab1, tab2, tab3 = st.tabs(["🗺️ İnteraktif Harita", "📈 Grafik Analizi", "🔮 Akıllı Tahmin"])

    with tab1:
        st.subheader(f"📍 {secilen_gün} - Saat {secilen_saat:02d}:00")
        h_veri = saatlik_veri[['lat', 'lon', 'Renk', 'Hiz', 'Arac']].copy()
        view = pdk.ViewState(latitude=41.0082, longitude=28.9784, zoom=10, pitch=45)
        layer = pdk.Layer("ScatterplotLayer", data=h_veri, get_position='[lon, lat]',
                          get_color='Renk', get_radius=nokta_boyu, pickable=True)
        st.pydeck_chart(pdk.Deck(initial_view_state=view, layers=[layer], tooltip={"text": "Hız: {Hiz} km/s"}))

    with tab2:
        st.subheader("📊 Bölgesel ve Zamansal İstatistikler")
        c1, c2 = st.columns(2)

        with c1:
            # 1. HIZ TRENDİ (Çizgi Grafik + Seçilen Saat İşaretçisi)
            trend = df_bolge.groupby('Saat')['Hiz'].mean().reset_index().sort_values('Saat')
            trend['Hiz'] = trend['Hiz'].round(1)  # Küsuratları temizle

            fig = px.line(trend, x='Saat', y='Hiz',
                          title=f"{secilen_yaka} - 24 Saatlik Hız Akışı",
                          labels={'Hiz': 'Hız (km/s)', 'Saat': 'Saat'},
                          markers=True)

            # Grafik (Çizgi kalınlığı ve renk)
            fig.update_traces(line=dict(width=3, color='#3498db'), mode='lines+markers',
                              hovertemplate="Saat: %{x}:00<br>Hiz: %{y} km/s")

            # X Eksenini 0-23 Arasına Sabitleme (Saatlerin kaymasını önler)
            fig.update_xaxes(tickmode='linear', dtick=1, range=[-0.5, 23.5])
            fig.update_yaxes(range=[0, trend['Hiz'].max() + 20])

            # --- SEÇİLEN SAATİ İŞARETLE ---
            if secilen_saat in trend['Saat'].values:
                # Seçilen saatin o anki hız değerini buluyoruz
                anlik_hiz = trend.loc[trend['Saat'] == secilen_saat, 'Hiz'].values[0]

                # Grafiğe parlak bir kırmızı noktası ekliyoruz
                fig.add_scatter(
                    x=[secilen_saat],
                    y=[anlik_hiz],
                    mode='markers',
                    marker=dict(color='red', size=15, symbol='diamond', line=dict(width=2, color='white')),
                    name=f"Şu an: {secilen_saat:02d}:00",
                    showlegend=True
                )

            st.plotly_chart(fig, use_container_width=True, key="hiz_trend_fixed")

        with c2:
            # 2. YOĞUNLUK DAĞILIMI (Bar Grafik)
            if not saatlik_veri.empty:
                ozet = saatlik_veri['Durum'].value_counts().reset_index()
                fig_bar = px.bar(ozet, x='Durum', y='count', color='Durum',
                                 title=f"Saat {secilen_saat:02d}:00 Trafik Dağılımı",
                                 color_discrete_map={"🚨 Yoğun": "red", "🟡 Akıcı": "gold", "🟢 Açık": "green"})
                st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart_fixed")
            else:
                st.warning("Bu saat diliminde seçilen kriterlere uygun veri noktası yok.")

    with tab3:
        st.header("🤖 Trafik Hızı Tahmin Asistanı")
        ai_model = model_egit(df)

        col1, col2 = st.columns([1, 2])
        with col1:
            t_tarih = st.date_input("Tahmin Tarihi:", datetime.now())
            t_saat = st.slider("Saat:", 0, 23, 18, key="t_slider")
            t_ilce = st.selectbox("Konum (İlçe):", list(ILCE_KOORDINAT.keys()))

            # --- ✅ LOKAL ARAÇ SAYISI ANALİZİ ---
            lat, lon = ILCE_KOORDINAT[t_ilce]
            # Seçilen konuma yakın olan verileri filtrele (0.02 derece tolerans)
            lokal_gecmis = df[
                (df['lat'].between(lat - 0.02, lat + 0.02)) &
                (df['lon'].between(lon - 0.02, lon + 0.02)) &
                (df['Saat'] == t_saat) &
                (df['Gun_No'] == t_tarih.weekday())
            ]

            # Eğer lokal veri yoksa genel ortalamaya dön (Hata önleyici)
            if not lokal_gecmis.empty:
                t_arac_otomatik = int(lokal_gecmis['Arac'].mean())
                gecmis_hiz_ort = lokal_gecmis['Hiz'].mean()
            else:
                t_arac_otomatik = int(df[df['Saat'] == t_saat]['Arac'].mean())
                gecmis_hiz_ort = df[df['Saat'] == t_saat]['Hiz'].mean()

            st.info(f"💡 {t_ilce} için bu saatte normalde **{t_arac_otomatik}** araç beklenir.")
            t_arac = st.number_input("Araç Sayısı (Düzenlenebilir):", value=t_arac_otomatik)

        with col2:
            bayram_mi = 1 if t_tarih in tr_holidays else 0
            bayram_adi = tr_holidays.get(t_tarih) if bayram_mi else "Normal Gün"

            # --- 🚀 KRİTİK DÜZELTME: 'Ay' PARAMETRESİ EKLENDİ ---
            # Model 7 sütun bekliyor, t_tarih.month bilgisini ekliyoruz.
            girdi = pd.DataFrame(
                [[t_saat, t_tarih.weekday(), t_arac, lat, lon, bayram_mi, t_tarih.month]],
                columns=['Saat', 'Gun_No', 'Arac', 'lat', 'lon', 'Is_Holiday', 'Ay']
            )
            
            tahmin = ai_model.predict(girdi)[0]

            st.subheader(f"🔮 {t_ilce} Analiz Raporu")
            st.metric(f"Tahmini Hız", f"{tahmin:.1f} km/s")

            # --- ✅ AKILLI KIYASLAMA MANTIĞI ---
            if bayram_mi:
                st.warning(
                    f"🎊 {bayram_adi} trafiği analiz ediliyor. Resmi tatil günlerinde kaza/aksaklık analizi devre dışı bırakıldı.")
            else:
                fark = tahmin - gecmis_hiz_ort
                if fark < -10:
                    st.error(
                        f"🚨 **Aksaklık Olasılığı:** Bu noktada normalde hız {gecmis_hiz_ort:.1f} km/s olurdu. Tahmin edilen {tahmin:.1f} km/s değeri bir aksaklığa veya yol çalışması gibi sebeplere işaret edebilir.")
                elif abs(fark) <= 10:
                    st.info(
                        f"📅 **Normal Seyir:** {t_ilce} için {t_saat:02d}:00 trafiği, geçmiş verilerdeki {gecmis_hiz_ort:.1f} km/s ortalamasıyla paralel görünüyor.")
                else:
                    st.success(f"✅ **Akıcı:** Yol bugün normalden ({gecmis_hiz_ort:.1f} km/s) daha açık görünüyor!")
else:
    st.info("👋 Analize başlamak için lütfen .csv dosyasını yükleyin.")
