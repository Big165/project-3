iimport streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# 🌟 โหลดข้อมูลตัวอย่าง (ใช้ชุดข้อมูล Sentiment Analysis)
@st.cache_data
def load_data():
    data = {
        "text": [
            "I love this product, it's amazing!",
            "This is the worst thing I have ever bought.",
            "Absolutely fantastic! Highly recommended.",
            "Terrible quality, I hate it.",
            "I'm very happy with this purchase.",
            "Not worth the money, very disappointed.",
            "Best experience ever, would buy again!",
            "Horrible, would not recommend.",
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
    }
    return pd.DataFrame(data)

df = load_data()

# 🌟 สร้างโมเดล Naïve Bayes
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# 🌟 อินเทอร์เฟซของ Streamlit
st.title("📢 Sentiment Analysis with Naïve Bayes")
st.write("ใส่ข้อความเพื่อดูว่ามีอารมณ์เชิงบวกหรือเชิงลบ")

# รับอินพุตจากผู้ใช้
user_input = st.text_area("💬 ป้อนข้อความที่ต้องการวิเคราะห์")

if user_input:
    prediction = model.predict([user_input])[0]
    sentiment = "😊 Positive" if prediction == 1 else "😡 Negative"
    st.subheader(f"🔍 ผลลัพธ์: {sentiment}")

st.markdown("---")
st.write("🔹 โมเดลนี้ใช้ Naïve Bayes และ TF-IDF Vectorizer สำหรับการวิเคราะห์ข้อความ")
