import os
import base64
from pathlib import Path

import streamlit as st
import google.generativeai as genai

# =========================
# Configuração do app
# =========================
st.set_page_config(page_title="Ensina Feridas (Gemini)", layout="centered")

st.markdown(
    """
    <style>
      /* Enxuga o topo do Streamlit */
      header {visibility: hidden;}
      [data-testid="stToolbar"] {visibility: hidden;}
      [data-testid="stHeader"] {display:none;}
      [data-testid="stDecoration"] {display:none;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}

      /* Menos padding no topo */
      .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 1.2rem !important;
      }

      /* Título menor */
      h1 { font-size: 1.6rem !important; margin-top: 0.2rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)


banner_path = Path(__file__).parent / "assets" / "banner-ensina-feridas.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)

# Logo (opcional) — coloque um arquivo logo.png na mesma pasta do app.py
# logo_path = Path(__file__).parent / "assets" / "logo.png"
# if logo_path.exists():
#     b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
#     st.markdown(f'<img src="data:image/png;base64,{b64}">', unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center; margin:0.25rem 0 0.25rem 0;'>🩹 Ensina Feridas – PET G10 UFPel</h2>", unsafe_allow_html=True)
st.caption("Streamlit + Gemini (SDK estável: `google-generativeai`).")

# =========================
# "GEM" educacional = system prompt forte + modelo base
# =========================
CLINICAL_HINT = (
    "Você é um especialista em feridas crônicas e protocolos de cuidado (TIME/TIMERS). "
    "Responda com orientação clínica segura e prática. "
    "Se faltarem dados, faça perguntas objetivas. "
    "Evite prescrever doses/condutas de alto risco sem contexto clínico. "
    "Quando houver sinais de alarme (ex.: infecção sistêmica, isquemia grave, dor desproporcional), "
    "recomende avaliação presencial."
)

EDU_HINT = (
    "Você é um especialista em ensino & aprendizagem no ensino superior (tutor). "
    "Seu objetivo é ensinar, não só responder. "
    "Use explicação progressiva (do básico ao avançado), exemplos, analogias e perguntas diagnósticas. "
    "Aplique metodologias ativas (PBL): formule hipóteses, peça dados que faltam e estimule raciocínio. "
    "Sempre que possível, devolva um mini-roteiro de estudo + um exercício curto com gabarito comentado. "
    "Mantenha o foco em feridas crônicas e protocolos TIME/TIMERS, com segurança clínica."
)

mode = st.radio("Modo", ["Ensino (tutor)", "Clínico (objetivo)"], horizontal=True)
SYSTEM_HINT = EDU_HINT if mode == "Ensino (tutor)" else CLINICAL_HINT

# =========================
# Pega API key (secrets > env)
# Aceita GOOGLE_API_KEY (preferido) ou GEMINI_API_KEY (legado)
# =========================
def get_api_key() -> str | None:
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            v = st.secrets.get(k)
        except Exception:
            v = None
        if v:
            return str(v).strip()
    return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip() or None


api_key = get_api_key()

if not api_key:
    st.error("Faltou a chave da API. Defina GOOGLE_API_KEY ou GEMINI_API_KEY.")
    st.info(
        "No Streamlit Cloud: Settings → Secrets\n\n"
        'GOOGLE_API_KEY = "SUA_CHAVE_AQUI"\n'
    )
    st.stop()

# Configura SDK estável
genai.configure(api_key=api_key)

# =========================
# Modelos disponíveis (cache)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def list_generate_models(_api_key_fingerprint: str) -> list[str]:
    """
    Lista modelos que suportam geração de conteúdo.
    No SDK `google-generativeai`, o nome costuma vir como `models/<nome>`.
    """
    models: list[str] = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" in methods and getattr(m, "name", None):
                models.append(m.name)
    except Exception:
        models = []
    return models or [
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
    ]


available_models = list_generate_models(api_key[:6] + "…" + api_key[-4:])

# =========================
# Controles do usuário
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    model_name = st.selectbox("Modelo", available_models, index=0)

with col2:
    temperature = st.slider(
        "Estilo da resposta",
        0.0,
        1.0,
        0.25 if mode == "Ensino (tutor)" else 0.3,
        0.05,
    )
    st.caption("Mais baixo = respostas objetivas • Mais alto = respostas mais explicativas")

prompt = st.text_area(
    "Pergunta / caso",
    height=220,
    placeholder=(
        "Ex.: Explique TIME/TIMERS e me faça 3 perguntas para avaliar se entendi.\n"
        "ou: Paciente com úlcera venosa há 8 meses, exsudato moderado, bordas maceradas..."
    ),
)

# =========================
# Execução
# =========================
def build_prompt(user_text: str) -> str:
    teaching_rules = ""
    if mode == "Ensino (tutor)":
        teaching_rules = (
            "\nFORMATO (modo ensino):"
            "\n1) Resposta curta (2–5 linhas) para situar."
            "\n2) Explicação em passos (bullet points)."
            "\n3) Perguntas diagnósticas (3–5)."
            "\n4) Exercício rápido + gabarito comentado."
            "\n5) Alertas de segurança (se aplicável).\n"
        )

    return f"""INSTRUÇÕES (contexto):
{SYSTEM_HINT}

SOLICITAÇÃO DO USUÁRIO:
{user_text}

REGRAS GERAIS:
- Seja prático e didático.
- Se houver risco (ex.: sinais de infecção sistêmica, isquemia grave, dor desproporcional), recomende avaliação presencial.
{teaching_rules}
"""


if st.button("Enviar para o Gemini", type="primary"):
    if not prompt.strip():
        st.warning("Escreve algo antes. O modelo não lê pensamento (ainda). 😄")
        st.stop()

    with st.spinner("Gerando resposta..."):
        try:
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content(
                build_prompt(prompt),
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                ),
            )

            st.subheader("Resposta:")
            text = getattr(resp, "text", None)
            if text:
                st.write(text)
            else:
                st.write(resp)

        except Exception as e:
            st.error("Erro ao chamar o Gemini:")
            st.exception(e)

st.divider()
st.caption("Dica: um projeto = um .venv. E, se der erro estranho, reinicie o terminal/VS Code.")
