import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

# ══════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Analisis Sentimen MBG",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CSS STYLING
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #0e1117; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4466;
        margin: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sentiment badge */
    .badge-positif {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white; padding: 8px 20px; border-radius: 20px;
        font-weight: 700; font-size: 1.1rem; display: inline-block;
    }
    .badge-negatif {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white; padding: 8px 20px; border-radius: 20px;
        font-weight: 700; font-size: 1.1rem; display: inline-block;
    }
    .badge-netral {
        background: linear-gradient(135deg, #636e72, #74b9ff);
        color: white; padding: 8px 20px; border-radius: 20px;
        font-weight: 700; font-size: 1.1rem; display: inline-block;
    }
    
    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 15px;
        text-align: center; margin-bottom: 25px;
    }
    .hero-title {
        font-size: 2rem; font-weight: 800; color: white; margin: 0;
    }
    .hero-subtitle {
        font-size: 1rem; color: rgba(255,255,255,0.85); margin-top: 8px;
    }
    
    /* Prob bar */
    .prob-row {
        display: flex; align-items: center; margin: 8px 0; gap: 10px;
    }
    .prob-label { width: 80px; color: #cdd6f4; font-size: 0.9rem; }
    .prob-bar-bg {
        flex: 1; background: #2d3250; border-radius: 10px; height: 22px; overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%; border-radius: 10px;
        display: flex; align-items: center; padding-left: 8px;
        font-size: 0.8rem; font-weight: 600; color: white;
        transition: width 0.5s ease;
    }
    
    /* Sidebar */
    .css-1d391kg { background: #1a1b2e; }
    
    /* Info box */
    .info-box {
        background: #1e2130; border-left: 4px solid #667eea;
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════════
BERT_MODEL  = 'indolem/indobert-base-uncased'
MAX_LEN     = 128
LABEL2ID    = {'positif': 0, 'netral': 1, 'negatif': 2}
ID2LABEL    = {0: 'positif', 1: 'netral', 2: 'negatif'}
COLORS      = {'positif': '#00b894', 'netral': '#74b9ff', 'negatif': '#e17055'}

slang_dict = {
    'gak':'tidak','ga':'tidak','ngga':'tidak','nggak':'tidak','gk':'tidak',
    'tdk':'tidak','tak':'tidak','blm':'belum','blum':'belum',
    'udah':'sudah','udh':'sudah','dah':'sudah',
    'bgt':'banget','bngt':'banget','yg':'yang','dgn':'dengan',
    'utk':'untuk','krn':'karena','sdh':'sudah','jd':'jadi',
    'tp':'tapi','klo':'kalau','kl':'kalau','kalo':'kalau',
    'ttg':'tentang','sm':'sama','bs':'bisa','msh':'masih',
    'trs':'terus','jg':'juga','lg':'lagi','lbh':'lebih',
    'pd':'pada','dr':'dari','dlm':'dalam','dg':'dengan',
    'mbg':'makan bergizi gratis','pgm':'program',
    'wkwk':'','haha':'','hihi':'','nih':'',
    'deh':'','dong':'','sih':'','aja':'saja',
}

def normalize_slang(text):
    return ' '.join([slang_dict.get(w, w) for w in text.split()])

def clean_for_bert(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300] if text.strip() else 'tidak ada teks'

# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════
class IndoBERTBiLSTM(nn.Module):
    def __init__(self, bert_model_name, hidden_size=256, num_layers=2,
                 num_classes=3, dropout=0.3):
        super(IndoBERTBiLSTM, self).__init__()
        self.bert   = AutoModel.from_pretrained(bert_model_name)
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_size * 2, hidden_size)
        self.relu      = nn.ReLU()
        self.fc2       = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out        = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state
        lstm_out, _     = self.bilstm(sequence_output)
        attn_weights    = torch.softmax(self.attention(lstm_out), dim=1)
        context         = (attn_weights * lstm_out).sum(dim=1)
        out = self.dropout(context)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ══════════════════════════════════════════════════════════════
# LOAD MODEL (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    
    # Load config
    if os.path.exists('model_config.pkl'):
        with open('model_config.pkl', 'rb') as f:
            cfg = pickle.load(f)
    else:
        cfg = {
            'bert_model_name': BERT_MODEL,
            'hidden_size': 256, 'num_layers': 2,
            'num_classes': 3, 'dropout': 0.3,
            'max_len': 128
        }

    # Load tokenizer
    tok_path = 'tokenizer' if os.path.exists('tokenizer') else cfg.get('bert_model_name', BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    # Build model
    model = IndoBERTBiLSTM(
        bert_model_name=cfg.get('bert_model_name', BERT_MODEL),
        hidden_size=cfg.get('hidden_size', 256),
        num_layers=cfg.get('num_layers', 2),
        num_classes=cfg.get('num_classes', 3),
        dropout=cfg.get('dropout', 0.3)
    ).to(device)

    # Load weights
    if os.path.exists('best_model.pt'):
        state = torch.load('best_model.pt', map_location=device)
        model.load_state_dict(state)
    
    model.eval()
    return model, tokenizer, device, cfg

@st.cache_data
def load_data():
    if os.path.exists('df_original.csv'):
        return pd.read_csv('df_original.csv')
    return None

@st.cache_data
def load_eval():
    if os.path.exists('eval_results.pkl'):
        with open('eval_results.pkl', 'rb') as f:
            return pickle.load(f)
    return None

# ══════════════════════════════════════════════════════════════
# FUNGSI PREDIKSI
# ══════════════════════════════════════════════════════════════
def predict_sentiment(text, model, tokenizer, device, max_len=128):
    clean = clean_for_bert(text)
    clean = normalize_slang(clean)
    enc   = tokenizer(
        clean, return_tensors='pt', truncation=True,
        max_length=max_len, padding='max_length'
    )
    with torch.no_grad():
        out   = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    pred = ID2LABEL[int(probs.argmax())]
    return pred, probs, clean

# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGASI
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 15px 0;'>
        <div style='font-size:2.5rem;'>🍽️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#667eea;'>Sentimen MBG</div>
        <div style='font-size:0.75rem; color:#8892b0;'>IndoBERT + BiLSTM</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    page = st.radio(
        "📋 Navigasi",
        ["🏠 Dashboard", "🔮 Prediksi Sentimen",
         "📊 Evaluasi Model", "☁️ WordCloud", "📁 Data Explorer"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("""
    <div class='info-box'>
        <div style='font-size:0.8rem; color:#8892b0;'>
        <b style='color:#cdd6f4;'>Arsitektur Model</b><br>
        IndoBERT → BiLSTM → Attention → Softmax
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════
model_loaded = False
try:
    with st.spinner("🔄 Loading model IndoBERT + BiLSTM..."):
        model, tokenizer, device, cfg = load_model()
    model_loaded = True
except Exception as e:
    st.sidebar.error(f"⚠️ Model belum tersedia\n\n{str(e)[:100]}")

df       = load_data()
eval_res = load_eval()

# ══════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>🍽️ Analisis Sentimen Program MBG</div>
        <div class='hero-subtitle'>
            Perbandingan Sentimen Masyarakat terhadap Program Makan Bergizi Gratis<br>
            menggunakan Deep Learning IndoBERT + BiLSTM pada Data Twitter/X
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        total = len(df)
        counts = df['sentiment'].value_counts() if 'sentiment' in df.columns else {}
        pos = counts.get('positif', 0)
        net = counts.get('netral', 0)
        neg = counts.get('negatif', 0)

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Tweet</div>
                <div class='metric-value' style='color:#667eea;'>{total:,}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>😊 Positif</div>
                <div class='metric-value' style='color:#00b894;'>{pos:,}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>😐 Netral</div>
                <div class='metric-value' style='color:#74b9ff;'>{net:,}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>😠 Negatif</div>
                <div class='metric-value' style='color:#e17055;'>{neg:,}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Charts
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("📊 Distribusi Label Sentimen")
            fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e2130')
            ax.set_facecolor('#1e2130')
            labels_list = ['positif', 'netral', 'negatif']
            values_list = [pos, net, neg]
            bar_colors  = [COLORS[l] for l in labels_list]
            bars = ax.bar(labels_list, values_list, color=bar_colors,
                          edgecolor='#2d3250', linewidth=2, width=0.5)
            for bar, v in zip(bars, values_list):
                ax.text(bar.get_x() + bar.get_width()/2, v + total*0.005,
                        f'{v:,}', ha='center', fontweight='bold',
                        color='white', fontsize=11)
            ax.set_ylabel('Jumlah Tweet', color='#8892b0')
            ax.tick_params(colors='#cdd6f4')
            ax.spines[['top','right','left','bottom']].set_color('#3d4466')
            ax.yaxis.label.set_color('#8892b0')
            for spine in ax.spines.values(): spine.set_color('#3d4466')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.subheader("🥧 Proporsi Sentimen")
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
            ax2.set_facecolor('#1e2130')
            if sum(values_list) > 0:
                wedges, texts, autotexts = ax2.pie(
                    values_list, labels=labels_list,
                    colors=bar_colors, autopct='%1.1f%%',
                    startangle=90, pctdistance=0.75,
                    wedgeprops={'linewidth': 2, 'edgecolor': '#1e2130'}
                )
                for t in texts: t.set_color('#cdd6f4')
                for at in autotexts:
                    at.set_color('white'); at.set_fontweight('bold')
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        # Akurasi model
        if eval_res:
            st.divider()
            st.subheader("🎯 Performa Model")
            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                ("Accuracy",  eval_res.get('accuracy', 0),  "#667eea"),
                ("Precision", eval_res.get('precision', 0), "#00b894"),
                ("Recall",    eval_res.get('recall', 0),    "#fdcb6e"),
                ("F1-Score",  eval_res.get('f1', 0),        "#e17055"),
            ]
            for col, (name, val, color) in zip([m1,m2,m3,m4], metrics):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{name}</div>
                        <div class='metric-value' style='color:{color};'>{val*100:.2f}%</div>
                    </div>""", unsafe_allow_html=True)

        # Sample tweets
        st.divider()
        st.subheader("💬 Contoh Tweet per Sentimen")
        tab1, tab2, tab3 = st.tabs(["😊 Positif", "😐 Netral", "😠 Negatif"])
        for tab, sent in zip([tab1, tab2, tab3], ['positif', 'netral', 'negatif']):
            with tab:
                subset = df[df['sentiment'] == sent].head(5) if 'sentiment' in df.columns else pd.DataFrame()
                if not subset.empty:
                    for _, row in subset.iterrows():
                        tweet = str(row.get('full_text', ''))[:200]
                        conf  = row.get('confidence', 0)
                        st.markdown(f"""
                        <div style='background:#1e2130; border-left:3px solid {COLORS[sent]};
                             padding:10px 14px; margin:6px 0; border-radius:0 8px 8px 0;'>
                            <span style='color:#cdd6f4; font-size:0.9rem;'>{tweet}</span>
                            <span style='color:#8892b0; font-size:0.75rem; float:right;'>
                                conf: {conf:.2f}</span>
                        </div>""", unsafe_allow_html=True)
    else:
        st.info("📂 Upload `df_original.csv` ke direktori app untuk melihat statistik dataset.")

# ══════════════════════════════════════════════════════════════
# PAGE 2: PREDIKSI SENTIMEN
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Sentimen":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>🔮 Prediksi Sentimen Real-Time</div>
        <div class='hero-subtitle'>Masukkan teks tweet untuk dianalisis menggunakan IndoBERT + BiLSTM</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("❌ Model belum berhasil dimuat. Pastikan file `best_model.pt` dan `tokenizer/` tersedia.")
        st.stop()

    # Contoh kalimat
    st.markdown("**💡 Atau coba contoh kalimat:**")
    examples = [
        "Program makan bergizi gratis sangat membantu anak sekolah Indonesia",
        "MBG cuma pencitraan saja, makanannya tidak layak makan",
        "Hari ini jadwal pembagian makan siang di sekolah",
        "Alhamdulillah anak saya dapat makan gratis setiap hari dari MBG",
        "Program MBG buang-buang anggaran, lebih baik untuk infrastruktur",
    ]
    cols = st.columns(len(examples))
    selected = ""
    for col, ex in zip(cols, examples):
        with col:
            if st.button(ex[:35]+"...", use_container_width=True, key=ex):
                selected = ex

    # Input teks
    user_input = st.text_area(
        "✍️ Masukkan teks tweet:",
        value=selected,
        height=120,
        placeholder="Ketik atau tempel tweet di sini...",
        key="tweet_input"
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        analyze = st.button("🔍 Analisis", type="primary", use_container_width=True)
    with col_btn2:
        clear = st.button("🗑️ Hapus", use_container_width=True)
        if clear:
            st.rerun()

    if analyze and user_input.strip():
        with st.spinner("🧠 Model sedang menganalisis..."):
            pred, probs, cleaned = predict_sentiment(
                user_input, model, tokenizer, device, cfg.get('max_len', 128)
            )

        st.divider()

        # Hasil prediksi utama
        emoji = {'positif': '✅', 'netral': '😐', 'negatif': '❌'}[pred]
        badge_class = f'badge-{pred}'
        conf_pct = probs.max() * 100

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown(f"""
            <div style='background:#1e2130; border-radius:15px; padding:25px; text-align:center;'>
                <div style='font-size:4rem;'>{emoji}</div>
                <div class='{badge_class}' style='margin-top:10px;'>{pred.upper()}</div>
                <div style='color:#8892b0; margin-top:10px; font-size:0.85rem;'>
                    Keyakinan model: <b style='color:white;'>{conf_pct:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown("**📊 Probabilitas per Kelas:**")
            for i, (label, color) in enumerate([
                ('positif', '#00b894'),
                ('netral',  '#74b9ff'),
                ('negatif', '#e17055')
            ]):
                pct = probs[i] * 100
                st.markdown(f"""
                <div class='prob-row'>
                    <div class='prob-label'>{label}</div>
                    <div class='prob-bar-bg'>
                        <div class='prob-bar-fill' style='width:{pct:.1f}%; background:{color};'>
                            {pct:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='info-box' style='margin-top:15px;'>
                <span style='color:#8892b0; font-size:0.8rem;'>Teks setelah preprocessing:</span><br>
                <span style='color:#cdd6f4; font-size:0.85rem;'><i>{cleaned[:150]}...</i></span>
            </div>
            """, unsafe_allow_html=True)

        # Analisis batch
        st.divider()
        st.subheader("📋 Analisis Multi-Tweet (Batch)")
        batch_input = st.text_area(
            "Masukkan beberapa tweet (pisahkan dengan baris baru):",
            height=150,
            placeholder="Tweet 1\nTweet 2\nTweet 3\n..."
        )
        if st.button("🔍 Analisis Semua", type="secondary"):
            tweets = [t.strip() for t in batch_input.split('\n') if t.strip()]
            if tweets:
                results = []
                prog = st.progress(0)
                for i, tw in enumerate(tweets):
                    p, prb, _ = predict_sentiment(tw, model, tokenizer, device, cfg.get('max_len', 128))
                    results.append({
                        'Tweet': tw[:80] + '...' if len(tw) > 80 else tw,
                        'Sentimen': p,
                        'Confidence': f"{prb.max()*100:.1f}%",
                        'Positif': f"{prb[0]*100:.1f}%",
                        'Netral':  f"{prb[1]*100:.1f}%",
                        'Negatif': f"{prb[2]*100:.1f}%",
                    })
                    prog.progress((i+1)/len(tweets))
                df_res = pd.DataFrame(results)
                st.dataframe(df_res, use_container_width=True)
                st.download_button(
                    "⬇️ Download Hasil CSV",
                    df_res.to_csv(index=False).encode('utf-8'),
                    "hasil_prediksi.csv", "text/csv"
                )
    elif analyze:
        st.warning("⚠️ Masukkan teks terlebih dahulu!")

# ══════════════════════════════════════════════════════════════
# PAGE 3: EVALUASI MODEL
# ══════════════════════════════════════════════════════════════
elif page == "📊 Evaluasi Model":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>📊 Evaluasi Model IndoBERT + BiLSTM</div>
        <div class='hero-subtitle'>Hasil performa model pada test set</div>
    </div>
    """, unsafe_allow_html=True)

    if eval_res is None:
        st.info("📂 File `eval_results.pkl` belum tersedia. Jalankan training terlebih dahulu.")
        st.stop()

    # Metrik utama
    st.subheader("🎯 Metrik Evaluasi")
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics_data = [
        ("Accuracy",   eval_res.get('accuracy', 0),  "#667eea"),
        ("Precision",  eval_res.get('precision', 0), "#00b894"),
        ("Recall",     eval_res.get('recall', 0),    "#fdcb6e"),
        ("F1-Score",   eval_res.get('f1', 0),        "#e17055"),
        ("Macro-AUC",  eval_res.get('macro_auc', 0), "#a29bfe"),
    ]
    for col, (name, val, color) in zip([m1,m2,m3,m4,m5], metrics_data):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{name}</div>
                <div class='metric-value' style='color:{color};'>{val*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.subheader("🔲 Confusion Matrix")
        cm_data = eval_res.get('cm', None)
        if cm_data:
            cm_array    = np.array(cm_data)
            class_names = ['positif', 'netral', 'negatif']
            fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1e2130')
            ax.set_facecolor('#1e2130')
            sns.heatmap(
                cm_array, annot=True, fmt='d', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', linewidths=0.5,
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'}
            )
            ax.set_title('Confusion Matrix', color='white', fontweight='bold', pad=10)
            ax.set_xlabel('Prediksi', color='#8892b0')
            ax.set_ylabel('Aktual',   color='#8892b0')
            ax.tick_params(colors='#cdd6f4')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # Training History
    with col2:
        st.subheader("📈 Training History")
        history = eval_res.get('history', None)
        if history and 'train_loss' in history:
            epochs_ran = range(1, len(history['train_loss']) + 1)
            fig2, axes = plt.subplots(2, 1, figsize=(6, 5), facecolor='#1e2130')
            for ax in axes: ax.set_facecolor('#1e2130')

            axes[0].plot(epochs_ran, history['train_loss'], 'b-o', label='Train', lw=2, ms=5)
            axes[0].plot(epochs_ran, history['val_loss'],   'r-o', label='Val',   lw=2, ms=5)
            axes[0].set_title('Loss', color='white', fontsize=10)
            axes[0].legend(labelcolor='white', facecolor='#2d3250')
            axes[0].tick_params(colors='#cdd6f4')
            axes[0].grid(alpha=0.2)
            for sp in axes[0].spines.values(): sp.set_color('#3d4466')

            axes[1].plot(epochs_ran, [a*100 for a in history['train_acc']], 'b-o', label='Train', lw=2, ms=5)
            axes[1].plot(epochs_ran, [a*100 for a in history['val_acc']],   'r-o', label='Val',   lw=2, ms=5)
            axes[1].set_title('Accuracy (%)', color='white', fontsize=10)
            axes[1].legend(labelcolor='white', facecolor='#2d3250')
            axes[1].tick_params(colors='#cdd6f4')
            axes[1].grid(alpha=0.2)
            for sp in axes[1].spines.values(): sp.set_color('#3d4466')

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        else:
            st.info("Training history tidak tersedia.")

    # Classification Report
    st.divider()
    st.subheader("📋 Classification Report Detail")
    report = eval_res.get('report', None)
    if report:
        rows = []
        for cls in ['positif', 'netral', 'negatif']:
            if cls in report:
                r = report[cls]
                rows.append({
                    'Kelas'     : cls.capitalize(),
                    'Precision' : f"{r.get('precision', 0)*100:.2f}%",
                    'Recall'    : f"{r.get('recall', 0)*100:.2f}%",
                    'F1-Score'  : f"{r.get('f1-score', 0)*100:.2f}%",
                    'Support'   : int(r.get('support', 0)),
                })
        if rows:
            df_report = pd.DataFrame(rows)
            st.dataframe(df_report, use_container_width=True, hide_index=True)

    # Arsitektur
    st.divider()
    st.subheader("🏗️ Arsitektur Model")
    st.markdown("""
    <div style='background:#1e2130; border-radius:12px; padding:20px; font-family:monospace;'>
    <pre style='color:#cdd6f4; margin:0;'>
    Tweet Input
         │
    ┌────▼──────────────────────────────┐
    │   IndoBERT Encoder                │
    │   (indolem/indobert-base-uncased) │
    │   768-dim contextual embedding    │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   BiLSTM (2 layers, hidden=256)  │
    │   Bidirectional → output: 512-dim │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   Attention Mechanism             │
    │   Focus on sentiment keywords     │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   Dense(512→256) + ReLU           │
    │   Dropout(0.3)                    │
    │   Dense(256→3) + Softmax          │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │   Output: positif / netral /      │
    │           negatif                 │
    └───────────────────────────────────┘
    </pre>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4: WORDCLOUD
# ══════════════════════════════════════════════════════════════
elif page == "☁️ WordCloud":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>☁️ WordCloud Analisis Sentimen MBG</div>
        <div class='hero-subtitle'>Kata-kata yang paling sering muncul per kelas sentimen</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None or 'sentiment' not in df.columns:
        st.info("📂 Upload `df_original.csv` untuk melihat WordCloud.")
        st.stop()

    col_wc   = st.columns([1,1,1])
    wc_cfg   = [
        ('positif', 'Greens', '😊 Sentimen Positif'),
        ('netral',  'Blues',  '😐 Sentimen Netral'),
        ('negatif', 'Reds',   '😠 Sentimen Negatif'),
    ]

    text_col = 'clean_text' if 'clean_text' in df.columns else 'full_text'

    for col, (sent, cmap, title) in zip(col_wc, wc_cfg):
        with col:
            st.markdown(f"<h4 style='text-align:center;color:{COLORS[sent]};'>{title}</h4>",
                        unsafe_allow_html=True)
            corpus = ' '.join(df[df['sentiment']==sent][text_col].dropna().tolist())
            if corpus.strip():
                wc = WordCloud(
                    width=500, height=350,
                    background_color='#1e2130',
                    colormap=cmap,
                    max_words=100,
                    collocations=False,
                    prefer_horizontal=0.9
                ).generate(corpus)
                fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1e2130')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig, use_container_width=True)
                plt.close()
                cnt = len(df[df['sentiment']==sent])
                st.markdown(f"<p style='text-align:center;color:#8892b0;'>{cnt:,} tweet</p>",
                            unsafe_allow_html=True)
            else:
                st.info("Tidak ada data")

    # Top kata per sentimen
    st.divider()
    st.subheader("📊 Top 10 Kata per Sentimen")
    cols_top = st.columns(3)
    for col, (sent, color, _) in zip(cols_top, [
        ('positif', '#00b894', ''),
        ('netral',  '#74b9ff', ''),
        ('negatif', '#e17055', ''),
    ]):
        with col:
            corpus = ' '.join(df[df['sentiment']==sent][text_col].dropna())
            if corpus:
                from collections import Counter
                words = [w for w in corpus.split() if len(w) > 3]
                top10 = Counter(words).most_common(10)
                if top10:
                    df_top = pd.DataFrame(top10, columns=['Kata', 'Frekuensi'])
                    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#1e2130')
                    ax.set_facecolor('#1e2130')
                    ax.barh(df_top['Kata'][::-1], df_top['Frekuensi'][::-1],
                            color=color, alpha=0.85)
                    ax.set_title(f'Top Kata — {sent.capitalize()}',
                                 color='white', fontweight='bold')
                    ax.tick_params(colors='#cdd6f4')
                    for sp in ax.spines.values(): sp.set_color('#3d4466')
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 5: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📁 Data Explorer":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>📁 Data Explorer</div>
        <div class='hero-subtitle'>Jelajahi dataset tweet analisis sentimen MBG</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.info("📂 Upload `df_original.csv` untuk menjelajahi data.")
        st.stop()

    # Filter
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        sent_filter = st.multiselect(
            "🎭 Filter Sentimen",
            options=['positif', 'netral', 'negatif'],
            default=['positif', 'netral', 'negatif']
        )
    with col_f2:
        search = st.text_input("🔍 Cari kata kunci:", placeholder="Ketik kata kunci...")
    with col_f3:
        if 'confidence' in df.columns:
            conf_min = st.slider("📊 Min Confidence", 0.0, 1.0, 0.0, 0.05)
        else:
            conf_min = 0.0

    # Apply filter
    df_show = df.copy()
    if sent_filter and 'sentiment' in df.columns:
        df_show = df_show[df_show['sentiment'].isin(sent_filter)]
    if search:
        df_show = df_show[df_show['full_text'].astype(str).str.contains(search, case=False, na=False)]
    if 'confidence' in df.columns and conf_min > 0:
        df_show = df_show[df_show['confidence'] >= conf_min]

    st.markdown(f"**Menampilkan {len(df_show):,} dari {len(df):,} tweet**")

    # Kolom yang ditampilkan
    show_cols = ['full_text', 'sentiment']
    if 'confidence' in df.columns: show_cols.append('confidence')
    if 'created_at' in df.columns: show_cols.append('created_at')
    if 'username'   in df.columns: show_cols.append('username')

    df_display = df_show[show_cols].head(500).reset_index(drop=True)
    st.dataframe(
        df_display,
        use_container_width=True,
        height=450,
        column_config={
            'full_text'  : st.column_config.TextColumn("Tweet", width="large"),
            'sentiment'  : st.column_config.TextColumn("Sentimen", width="small"),
            'confidence' : st.column_config.NumberColumn("Confidence", format="%.3f"),
            'created_at' : st.column_config.TextColumn("Tanggal", width="medium"),
            'username'   : st.column_config.TextColumn("Username", width="medium"),
        }
    )

    # Download
    st.download_button(
        "⬇️ Download Data yang Difilter (CSV)",
        df_show.to_csv(index=False).encode('utf-8'),
        "data_filtered.csv", "text/csv"
    )

    # Statistik cepat
    st.divider()
    st.subheader("📊 Statistik Dataset")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if 'confidence' in df.columns:
            st.markdown("**Distribusi Confidence Score:**")
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='#1e2130')
            ax.set_facecolor('#1e2130')
            ax.hist(df['confidence'], bins=40, color='#667eea', edgecolor='#1e2130')
            ax.set_xlabel('Confidence', color='#8892b0')
            ax.set_ylabel('Frekuensi', color='#8892b0')
            ax.tick_params(colors='#cdd6f4')
            for sp in ax.spines.values(): sp.set_color('#3d4466')
            st.pyplot(fig, use_container_width=True)
            plt.close()
    with col_s2:
        st.markdown("**Statistik Panjang Tweet:**")
        df['tweet_len'] = df['full_text'].astype(str).str.len()
        stats = df['tweet_len'].describe()
        stat_df = pd.DataFrame({
            'Statistik': ['Rata-rata', 'Minimum', 'Maximum', 'Std Dev'],
            'Nilai': [f"{stats['mean']:.1f}", f"{stats['min']:.0f}",
                      f"{stats['max']:.0f}", f"{stats['std']:.1f}"]
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

