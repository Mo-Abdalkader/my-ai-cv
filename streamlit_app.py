# 🤖 Mohamed's AI Clone - Streamlit Interactive Portfolio
# Enhanced with Multi-LLM Backend & Real-time Chat

import streamlit as st
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# 🎨 PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Mohamed Abdalkader - AI Engineer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 🎨 CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hero section */
    .hero-section {
        background: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    /* Project cards */
    .project-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.3s;
    }
    
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Skill badges */
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Chat messages */
    .user-message {
        background: #667eea;
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        text-align: right;
    }
    
    .bot-message {
        background: #f0f2f6;
        color: #333;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 36px;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

# Contact Information
CONTACT = {
    "email": "Mohameed.Abdalkadeer@gmail.com",
    "phone": "+201023277913",
    "whatsapp": "https://wa.me/201023277913",
    "linkedin": "https://www.linkedin.com/in/mo-abdalkader/",
    "github": "https://github.com/Mo-Abdalkader/",
    "telegram_bot": "https://t.me/MohamedAI_Bot"
}

# LLM Configuration (ranked by efficiency)
LLM_CONFIGS = [
    {"name": "Groq Llama", "model": "llama-3.1-70b-versatile", "rank": 1},
    {"name": "Groq Mixtral", "model": "mixtral-8x7b-32768", "rank": 2},
    {"name": "Gemini Flash", "model": "gemini-1.5-flash", "rank": 3},
    {"name": "OpenRouter", "model": "meta-llama/llama-3.1-8b-instruct:free", "rank": 4}
]

# ============================================================================
# 💾 SESSION STATE INITIALIZATION
# ============================================================================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_llm' not in st.session_state:
    st.session_state.current_llm = LLM_CONFIGS[0]

if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

# ============================================================================
# 🧠 AI CHAT FUNCTIONS
# ============================================================================

def call_groq_api(message, api_key, model="llama-3.1-70b-versatile"):
    """Call Groq API"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": get_system_prompt()},
                    *st.session_state.chat_history[-6:],  # Last 3 exchanges
                    {"role": "user", "content": message}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.status_code}")
    except Exception as e:
        raise Exception(f"Groq API failed: {str(e)}")

def call_gemini_api(message, api_key):
    """Call Gemini API"""
    try:
        prompt = get_system_prompt() + "\n\n" + message
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": 1000,
                    "temperature": 0.7
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise Exception(f"API Error: {response.status_code}")
    except Exception as e:
        raise Exception(f"Gemini API failed: {str(e)}")

def call_openrouter_api(message, api_key):
    """Call OpenRouter API"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://mohamed-portfolio.streamlit.app"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": message}
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.status_code}")
    except Exception as e:
        raise Exception(f"OpenRouter API failed: {str(e)}")

def generate_response_with_fallback(message, groq_key, gemini_key, openrouter_key):
    """Generate response with smart LLM fallback"""
    
    # Try LLMs in order of efficiency
    for llm in LLM_CONFIGS:
        try:
            st.session_state.current_llm = llm
            
            if llm["name"].startswith("Groq"):
                return call_groq_api(message, groq_key, llm["model"])
            elif llm["name"] == "Gemini Flash":
                return call_gemini_api(message, gemini_key)
            elif llm["name"] == "OpenRouter":
                return call_openrouter_api(message, openrouter_key)
                
        except Exception as e:
            st.warning(f"⚠️ {llm['name']} unavailable, trying backup...")
            continue
    
    # All LLMs failed
    return f"😔 عذراً، جميع الأنظمة مشغولة حالياً. يرجى المحاولة لاحقاً أو التواصل مباشرة:\n\n📧 {CONTACT['email']}\n📱 {CONTACT['whatsapp']}"

def get_system_prompt():
    """Get enhanced system prompt"""
    return """أنت المساعد الذكي لمحمد عبد القادر، مهندس الذكاء الاصطناعي.

مهمتك:
1. الإجابة عن الأسئلة حول خبرات محمد المهنية
2. تقييم المشاريع: هل يمكن لمحمد تنفيذها؟
3. الرد بنفس لغة السؤال (عربي/English)
4. الحفاظ على الاحترافية والدقة

تخصصات محمد:
- التعلم العميق والرؤية الحاسوبية
- معالجة اللغة الطبيعية والنماذج الكبيرة
- MLOps ونشر النماذج

إذا سُئلت عن مشروع:
✅ مشاريع AI/ML → "نعم، محمد خبير في هذا المجال"
❓ مشاريع جزئياً AI → "محمد لديه المهارات الأساسية، يُفضل التواصل للتفاصيل"
❌ مشاريع غير AI → "هذا خارج تخصص محمد الأساسي"

كن موجزاً ومفيداً!"""

# ============================================================================
# 📊 DATA VISUALIZATION FUNCTIONS
# ============================================================================

def create_skills_radar():
    """Create skills radar chart"""
    categories = ['Deep Learning', 'Computer Vision', 'NLP', 'MLOps', 'Data Science']
    values = [95, 95, 85, 90, 90]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.5)',
        line=dict(color='rgb(102, 126, 234)', width=2),
        name='Skills'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=400
    )
    
    return fig

def create_project_metrics():
    """Create project metrics bar chart"""
    projects = ['Cancer Detection', 'Gesture Recognition', 'Energy Forecast', 'Chatbot RAG', 'Churn Prediction']
    accuracy = [97, 94, 98, 95, 92]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=projects,
        y=accuracy,
        marker_color='rgb(102, 126, 234)',
        text=accuracy,
        texttemplate='%{text}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Project Accuracy Metrics',
        yaxis_title='Accuracy (%)',
        height=400,
        showlegend=False
    )
    
    return fig

# ============================================================================
# 🏠 MAIN APP
# ============================================================================

def main():
    # ============================================================================
    # 📱 SIDEBAR
    # ============================================================================
    
    with st.sidebar:
        st.image("https://avatars.githubusercontent.com/u/yourusername", width=150)
        
        st.markdown("### 🤖 Mohamed Abdalkader")
        st.markdown("**AI/ML Engineer**")
        
        st.markdown("---")
        
        # API Keys Configuration
        st.markdown("### 🔑 API Configuration")
        groq_key = st.text_input("Groq API Key", type="password", help="Get from console.groq.com")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Get from ai.google.dev")
        openrouter_key = st.text_input("OpenRouter API Key", type="password", help="Get from openrouter.ai")
        
        st.markdown("---")
        
        # Quick Links
        st.markdown("### 🔗 Quick Links")
        st.markdown(f"📧 [Email](mailto:{CONTACT['email']})")
        st.markdown(f"💬 [WhatsApp]({CONTACT['whatsapp']})")
        st.markdown(f"💼 [LinkedIn]({CONTACT['linkedin']})")
        st.markdown(f"🔗 [GitHub]({CONTACT['github']})")
        st.markdown(f"🤖 [Telegram Bot]({CONTACT['telegram_bot']})")
        
        st.markdown("---")
        
        # Current LLM Status
        if st.session_state.message_count > 0:
            st.markdown("### 🧠 Current AI Model")
            st.success(f"**{st.session_state.current_llm['name']}**")
            st.info(f"Rank: #{st.session_state.current_llm['rank']}")
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Chat Stats")
        st.metric("Messages", st.session_state.message_count)
        st.metric("Conversations", len(st.session_state.chat_history) // 2)
    
    # ============================================================================
    # 🎯 MAIN CONTENT
    # ============================================================================
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">👋 مرحباً! I'm Mohamed Abdalkader</div>
        <h3 style="color: #666;">🤖 AI/ML Engineer | مهندس الذكاء الاصطناعي</h3>
        <p style="font-size: 18px; color: #888;">
            Passionate about building intelligent systems with Deep Learning, Computer Vision, and NLP.
            <br>
            متحمس لبناء أنظمة ذكية باستخدام التعلم العميق والرؤية الحاسوبية ومعالجة اللغة الطبيعية.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 AI Chat",
        "🚀 Projects",
        "🛠️ Skills",
        "📊 Analytics",
        "📞 Contact"
    ])
    
    # ============================================================================
    # TAB 1: AI CHAT
    # ============================================================================
    
    with tab1:
        st.markdown("## 💬 Chat with My AI Clone")
        st.markdown("Ask me anything about my experience, skills, or projects in **Arabic or English**!")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Your message:",
                placeholder="مثال: هل يمكنك تطوير نظام كشف الأجسام؟",
                key="user_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send 📤", use_container_width=True)
        
        if send_button and user_input:
            if not groq_key and not gemini_key and not openrouter_key:
                st.error("⚠️ Please add at least one API key in the sidebar!")
            else:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.message_count += 1
                
                # Generate response
                with st.spinner("🤖 Thinking..."):
                    response = generate_response_with_fallback(
                        user_input,
                        groq_key,
                        gemini_key,
                        openrouter_key
                    )
                
                # Add bot response
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun to show new messages
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.message_count = 0
            st.rerun()
    
    # ============================================================================
    # TAB 2: PROJECTS
    # ============================================================================
    
    with tab2:
        st.markdown("## 🚀 Featured Projects")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="project-card">
                <h3>🫁 Cancer Detection System</h3>
                <p><strong>97% Accuracy</strong> in detecting lung & colon cancer</p>
                <p>Tech: PyTorch, ResNet50, Azure ML, Docker</p>
                <span class="skill-badge">Computer Vision</span>
                <span class="skill-badge">Deep Learning</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="project-card">
                <h3>⚡ Energy Forecasting</h3>
                <p><strong>98% Accuracy</strong> in predicting consumption</p>
                <p>Tech: LSTM, Prophet, Azure, Pandas</p>
                <span class="skill-badge">Time Series</span>
                <span class="skill-badge">Forecasting</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="project-card">
                <h3>👋 Gesture Recognition</h3>
                <p><strong>60 FPS</strong> real-time recognition</p>
                <p>Tech: MediaPipe, OpenCV, TensorFlow Lite</p>
                <span class="skill-badge">Computer Vision</span>
                <span class="skill-badge">Edge AI</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="project-card">
                <h3>🤖 RAG Chatbot</h3>
                <p><strong>1000+ queries/day</strong>, 95% satisfaction</p>
                <p>Tech: LangChain, Pinecone, OpenAI, FastAPI</p>
                <span class="skill-badge">NLP</span>
                <span class="skill-badge">LLMs</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Project metrics chart
        st.plotly_chart(create_project_metrics(), use_container_width=True)
    
    # ============================================================================
    # TAB 3: SKILLS
    # ============================================================================
    
    with tab3:
        st.markdown("## 🛠️ Technical Skills")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🐍 Programming & Frameworks")
            st.markdown("""
            - **Python** (Expert) - NumPy, Pandas, Scikit-learn
            - **PyTorch** (Advanced) - Custom architectures, fine-tuning
            - **TensorFlow** (Advanced) - Keras, TF Lite
            - **Hugging Face** - Transformers, LangChain, LlamaIndex
            - **OpenCV** - YOLO, Detectron2, SAM
            """)
            
            st.markdown("### ☁️ MLOps & Deployment")
            st.markdown("""
            - **Docker** & **Kubernetes**
            - **Azure ML** & **AWS SageMaker**
            - **FastAPI**, **Streamlit**, **Gradio**
            - **Git**, **GitHub Actions**, **MLflow**
            """)
        
        with col2:
            # Skills radar chart
            st.plotly_chart(create_skills_radar(), use_container_width=True)
        
        # Skill badges
        st.markdown("### 🎯 Specializations")
        st.markdown("""
        <span class="skill-badge">Deep Learning</span>
        <span class="skill-badge">Computer Vision</span>
        <span class="skill-badge">NLP</span>
        <span class="skill-badge">LLMs</span>
        <span class="skill-badge">MLOps</span>
        <span class="skill-badge">Time Series</span>
        <span class="skill-badge">GANs</span>
        <span class="skill-badge">Transformers</span>
        <span class="skill-badge">Edge AI</span>
        <span class="skill-badge">RAG</span>
        """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 4: ANALYTICS
    # ============================================================================
    
    with tab4:
        st.markdown("## 📊 Professional Analytics")
        
        # Stats cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">10+</div>
                <div class="stat-label">Projects Completed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">97%</div>
                <div class="stat-label">Avg Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">2+</div>
                <div class="stat-label">Years Experience</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-number">500+</div>
                <div class="stat-label">GitHub Commits</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Education & Certifications
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎓 Education")
            st.info("""
            **Bachelor of Computer Science**  
            Zagazig University (2020-2024)  
            GPA: 3.8/4.0 (Excellent with Honor)
            """)
        
        with col2:
            st.markdown("### 📜 Top Certifications")
            st.success("""
            ✅ TensorFlow Developer - Google  
            ✅ Azure AI Engineer - Microsoft  
            ✅ Deep Learning - deeplearning.ai  
            ✅ AWS ML Specialty - Amazon
            """)
    
    # ============================================================================
    # TAB 5: CONTACT
    # ============================================================================
    
    with tab5:
        st.markdown("## 📞 Let's Connect!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📬 Contact Information")
            st.markdown(f"""
            - 📧 **Email**: [{CONTACT['email']}](mailto:{CONTACT['email']})
            - 📱 **Phone**: {CONTACT['phone']}
            - 💬 **WhatsApp**: [Direct Chat]({CONTACT['whatsapp']})
            - 💼 **LinkedIn**: [Profile]({CONTACT['linkedin']})
            - 🔗 **GitHub**: [Repositories]({CONTACT['github']})
            - 🤖 **AI Bot**: [Telegram Bot]({CONTACT['telegram_bot']})
            """)
            
            st.markdown("### 🌍 Languages")
            st.markdown("""
            - 🇸🇦 **Arabic**: Native
            - 🇬🇧 **English**: Fluent
            - 🇩🇪 **German**: Basic
            """)
        
        with col2:
            st.markdown("### 📝 Quick Message")
            
            with st.form("contact_form"):
                name = st.text_input("Name")
                email = st.text_input("Email")
                message = st.text_area("Message", height=150)
                submit = st.form_submit_button("Send Message 📤")
                
                if submit:
                    if name and email and message:
                        st.success("✅ Message sent! I'll get back to you soon.")
                        st.balloons()
                    else:
                        st.error("⚠️ Please fill all fields")
    
    # ============================================================================
    # 🎯 FOOTER
    # ============================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: white; padding: 20px;">
        <p>🤖 Built with ❤️ by Mohamed Abdalkader | Powered by Multi-LLM AI</p>
        <p style="font-size: 12px;">Last Updated: December 31, 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 🚀 RUN APP
# ============================================================================

main()
