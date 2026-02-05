import os
import base64
from pathlib import Path

import streamlit as st
import google.generativeai as genai

import io

# PDF (ReportLab) — usado para exportar A4 com banner no cabeçalho
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    _PDF_OK = True
except Exception:
    _PDF_OK = False

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


banner_path = Path(__file__).parent / "assets" / "banner.pdf.a4.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)
else:
    st.caption(f"Banner não encontrado: {banner_path}")

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

# =========================
# Exportação para PDF (A4) com banner no cabeçalho
# =========================
def wrap_text(text, max_chars=110):
    """Quebra texto em linhas por comprimento aproximado (robusto e simples)."""
    lines: list[str] = []
    for para in (text or "").splitlines():
        if not para.strip():
            lines.append("")
            continue
        words = para.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if len(test) <= max_chars:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
    return lines


def gerar_pdf_a4(pergunta: str, resposta: str) -> bytes:
    """Gera PDF A4 com banner (assets/banner.pdf.a4.png) no topo e texto abaixo."""

    if not _PDF_OK:
        raise RuntimeError("ReportLab não está disponível. Instale 'reportlab' no requirements.txt.")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    largura, altura = A4

    margem_esq = 2 * cm
    margem_dir = 2 * cm
    margem_inf = 2 * cm
    y = altura - 2 * cm

    banner = Path(__file__).parent / "assets" / "banner.pdf.a4.png"
    if banner.exists():
        img = ImageReader(str(banner))
        iw, ih = img.getSize()
        largura_util = largura - margem_esq - margem_dir
        escala = largura_util / float(iw)
        alt_img = float(ih) * escala

        c.drawImage(
            img,
            margem_esq,
            y - alt_img,
            width=largura_util,
            height=alt_img,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= alt_img + 0.8 * cm
    else:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margem_esq, y, f"Banner não encontrado: {banner}")
        y -= 0.8 * cm

    def _new_page():
        nonlocal y
        c.showPage()
        y = altura - 2 * cm

    # Título: Pergunta
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margem_esq, y, "Pergunta / Caso")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    for linha in wrap_text(pergunta):
        if y < margem_inf:
            _new_page()
            c.setFont("Helvetica", 10)
        c.drawString(margem_esq, y, linha)
        y -= 0.45 * cm

    y -= 0.7 * cm
    if y < margem_inf:
        _new_page()

    # Título: Resposta
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margem_esq, y, "Resposta do Sistema")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    for linha in wrap_text(resposta):
        if y < margem_inf:
            _new_page()
            c.setFont("Helvetica", 10)
        c.drawString(margem_esq, y, linha)
        y -= 0.45 * cm

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()



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
            final_text = text if text else str(resp)
            st.write(final_text)

            # Guarda para exportação em PDF
            st.session_state["ultima_pergunta"] = prompt
            st.session_state["ultima_resposta"] = final_text
            st.session_state["ultimo_modelo"] = model_name
            st.session_state["ultimo_modo"] = mode

            # Botão de exportação (fica logo após a resposta)
            st.markdown("### 📄 Exportar resposta")
            if not _PDF_OK:
                st.warning("Exportação PDF indisponível: instale `reportlab` no requirements.txt.")
            else:
                try:
                    pdf_bytes = gerar_pdf_a4(prompt, final_text)
                    st.download_button(
                        "Gerar PDF A4 (com banner)",
                        data=pdf_bytes,
                        file_name="ensina_feridas_resposta.pdf",
                        mime="application/pdf",
                        key="download_pdf_a4",
                    )
                except Exception as _e:
                    st.error("Falha ao gerar o PDF.")
                    st.exception(_e)

        except Exception as e:
            st.error("Erro ao chamar o Gemini:")
            st.exception(e)


st.divider()
st.caption("Dica: um projeto = um .venv. E, se der erro estranho, reinicie o terminal/VS Code.")
