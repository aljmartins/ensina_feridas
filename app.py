import os
import base64
from pathlib import Path
import io

import json
import re

import streamlit as st
import google.generativeai as genai

# PDF (ReportLab) — exportar A4 com banner, rodapé e numeração
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

# Enxuga o topo do Streamlit (remove espaço antes do banner)
st.markdown(
    """
    <style>
      header {visibility: hidden;}
      [data-testid="stToolbar"] {visibility: hidden;}
      [data-testid="stHeader"] {display:none;}
      [data-testid="stDecoration"] {display:none;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}

      .block-container {
        padding-top: 0.25rem !important;
        padding-bottom: 1.2rem !important;
      }
      h1 { font-size: 1.6rem !important; margin-top: 0.2rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Banner da página (UI)
banner_path = Path(__file__).parent / "assets" / "banner.pdf.a4.png"
if banner_path.exists():
    st.image(str(banner_path), use_container_width=True)
else:
    st.caption(f"Banner não encontrado: {banner_path}")

st.markdown(
    "<h2 style='text-align:center; margin:0.25rem 0 0.25rem 0;'>🩹 Ensina Feridas </h2>",
    unsafe_allow_html=True,
)
st.caption("Streamlit + Gemini (SDK estável: `google-generativeai`).")

from pathlib import Path
import base64
import streamlit as st

# --- Instagram icon ---
insta_path = Path("assets/instagram.png")
insta_b64 = base64.b64encode(insta_path.read_bytes()).decode()
enf_path = Path("assets/logo.enfermagem.png")
enf_b64 = base64.b64encode(enf_path.read_bytes()).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:12px; margin-top:-10px; margin-bottom:12px;">
        <img src="data:image/png;base64,{insta_b64}" width="24">
        <a href="https://www.instagram.com/amorapele_ufpel/" target="_blank"
           style="text-decoration:none; font-weight:500;">
           Amor à Pele
        </a>
        <span>|</span>
        <a href="https://www.instagram.com/g10petsaude/" target="_blank"
           style="text-decoration:none; font-weight:500;">
           PET G10
        </a>
        <span>|</span>
        <img src="data:image/png;base64,{enf_b64}" width="24">
        <a href="https://wp.ufpel.edu.br/fen/" target="_blank"
           style="text-decoration:none; font-weight:500;">
           Faculdade de Enfermagem – UFPel
        </a>
    </div>
    """,
    unsafe_allow_html=True
)




# =========================
# Prompts do "GEM"
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
# API key (secrets > env)
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

genai.configure(api_key=api_key)


# =========================
# Modelos disponíveis (cache)
# =========================
@st.cache_data(ttl=3600, show_spinner=False)
def list_generate_models(_api_key_fingerprint: str) -> list[str]:
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

auto_sketch = st.checkbox(
    "Sugerir esboço quando fizer sentido",
    value=True,
    help="O app decide automaticamente se um esboço/figura ajudaria na resposta e, se sim, gera um prompt pronto para você colar em um gerador de imagens.",
)

import streamlit.components.v1 as components

# =========================
# Caixa de pergunta com microfone
# =========================

# Microfone: injeta texto direto no textarea do Streamlit via DOM
components.html(
    """
    <style>
      .mic-row { display:flex; align-items:center; gap:10px; margin-bottom:2px; }
      .mic-btn {
        background:#0066cc; color:white; border:none;
        width:40px; height:40px; border-radius:50%;
        font-size:1.2rem; cursor:pointer; flex-shrink:0;
        display:flex; align-items:center; justify-content:center;
        transition:background .2s;
      }
      .mic-btn.listening { background:#e53935; animation:pulse 1s infinite; }
      @keyframes pulse {
        0%,100%{box-shadow:0 0 0 0 rgba(229,57,53,.35);}
        50%{box-shadow:0 0 0 8px rgba(229,57,53,0);}
      }
      #micStatus { font-size:.82rem; color:#555; }
    </style>
    <div class="mic-row">
      <button class="mic-btn" id="micBtn" title="Falar pergunta">🎤</button>
      <span id="micStatus">Toque em 🎤 para falar (Chrome/Edge) — o texto vai para a caixa abaixo</span>
    </div>
    <script>
    (function(){
      const btn    = document.getElementById('micBtn');
      const status = document.getElementById('micStatus');
      let listening = false;

      // Injeta texto no textarea do Streamlit que está na janela pai
      function fillStreamlitTextarea(text) {
        try {
          const doc = window.parent.document;
          // Streamlit renderiza textareas com data-testid="stTextArea" > textarea
          const ta = doc.querySelector('textarea[aria-label="Pergunta / caso"]')
                  || doc.querySelector('[data-testid="stTextArea"] textarea')
                  || doc.querySelector('textarea');
          if (!ta) { status.textContent = '⚠️ Caixa não encontrada — tente novamente.'; return; }
          // Força React a reconhecer a mudança
          const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype, 'value').set;
          nativeInputValueSetter.call(ta, text);
          ta.dispatchEvent(new Event('input', { bubbles: true }));
          ta.dispatchEvent(new Event('change', { bubbles: true }));
          ta.focus();
          status.textContent = '✅ Texto na caixa — edite se quiser e clique Enviar.';
        } catch(e) {
          status.textContent = '⚠️ Erro ao preencher: ' + e.message;
        }
      }

      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {
        status.textContent = '⚠️ Voz indisponível — use o 🎤 do teclado do celular.';
        btn.style.opacity = '.4';
        btn.addEventListener('click', () => alert('Use Chrome ou Edge para reconhecimento de voz.'));
        return;
      }

      const rec = new SR();
      rec.lang = 'pt-BR';
      rec.continuous = false;
      rec.interimResults = true;

      rec.onstart = () => {
        listening = true;
        btn.classList.add('listening');
        btn.innerHTML = '🔴';
        status.textContent = '🎙️ Ouvindo…';
      };
      rec.onresult = (e) => {
        let interim = '', final = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const t = e.results[i][0].transcript;
          if (e.results[i].isFinal) final += t; else interim += t;
        }
        if (final) { fillStreamlitTextarea(final.trim()); }
        else { status.textContent = '…' + interim; }
      };
      rec.onerror = (e) => {
        status.textContent = '❌ Erro: ' + e.error;
        btn.classList.remove('listening'); btn.innerHTML = '🎤'; listening = false;
      };
      rec.onend = () => {
        listening = false; btn.classList.remove('listening'); btn.innerHTML = '🎤';
      };
      btn.addEventListener('click', () => { if (listening) rec.stop(); else rec.start(); });
    })();
    </script>
    """,
    height=55,
    scrolling=False,
)

# Campo de texto nativo — editável normalmente (voz ou digitado)
prompt = st.text_area(
    "Pergunta / caso",
    height=200,
    placeholder=(
        "Conte sua dúvida ou situação do dia a dia.\n"
        "Ex.: Tenho pé diabético e amanhã vou a um casamento. Que sapato posso usar?"
    ),
    key="prompt_area",
)

# Botão nativo do Streamlit
enviar = st.button("🚀 Enviar para o Gemini", type="primary")


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
# Decisão: precisa de esboço?
# =========================
def decidir_esboco(pergunta: str, resposta: str, modelo_decisor: str | None = None) -> dict:
    """
    Usa o próprio Gemini para decidir se um esboço/figura ajuda, e propõe um prompt de imagem.
    Retorna dict com:
      - need_sketch: bool
      - reason: str
      - sketch_prompt: str
    """
    try:
        model_id = modelo_decisor or model_name
        model = genai.GenerativeModel(model_name=model_id)

        decisor_prompt = f"""
Você é um assistente que decide se um ESBOÇO/FIGURA simples ajudaria a resposta.
Contexto: o app é sobre feridas crônicas (TIME/TIMERS), mas a pergunta pode ser geral.

Responda SOMENTE em JSON válido, SEM markdown, SEM texto extra, no formato:
{{"need_sketch": true/false, "reason": "...", "sketch_prompt": "..."}}

Regras:
- need_sketch = true quando uma figura melhoraria MUITO a compreensão (ex.: anatomia, posicionamento, escolha de calçado/órtese, passo-a-passo de curativo, fluxogramas, comparação visual, layout de equipamento).
- need_sketch = false quando for pura explicação textual, listas simples, ou quando um desenho pode induzir erro clínico.
- Se need_sketch = false, deixe sketch_prompt como string vazia "".
- Se need_sketch = true, crie um prompt curto, bem específico, para gerar uma imagem didática, sem conteúdo chocante. Evite sangue explícito.

PERGUNTA:
{pergunta}

RESPOSTA (resumo):
{resposta[:1200]}
"""
        resp = model.generate_content(
            decisor_prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1),
        )
        raw = (getattr(resp, "text", None) or "").strip()

        # tenta JSON direto; se vier “com sujeira”, extrai o primeiro {...}
        try:
            data = json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            data = json.loads(m.group(0)) if m else {"need_sketch": False, "reason": "Não consegui interpretar a decisão.", "sketch_prompt": ""}

        return {
            "need_sketch": bool(data.get("need_sketch", False)),
            "reason": str(data.get("reason", "")).strip(),
            "sketch_prompt": str(data.get("sketch_prompt", "")).strip(),
        }
    except Exception:
        return {"need_sketch": False, "reason": "Falha ao decidir esboço.", "sketch_prompt": ""}

# =========================
# Exportação PDF
# =========================
def wrap_text(text: str, max_chars: int = 110) -> list[str]:
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


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, footer_text: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self._footer_text = footer_text

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for i, state in enumerate(self._saved_page_states, start=1):
            self.__dict__.update(state)
            self._draw_footer(i, num_pages)
            super().showPage()
        super().save()

    def _draw_footer(self, page_num: int, total_pages: int):
        width, _ = self._pagesize
        margin = 2 * cm
        y = 1.2 * cm

        self.setFont("Helvetica", 9)
        self.drawString(margin, y, self._footer_text)
        self.drawRightString(width - margin, y, f"Página {page_num} de {total_pages}")


def gerar_pdf_a4(pergunta: str, resposta: str) -> bytes:
    """Gera PDF A4 com banner no cabeçalho, rodapé fixo e numeração."""
    footer_text = "PET G10 UFPel - Telemonitoramento de Feridas Crônicas"

    buffer = io.BytesIO()
    c = NumberedCanvas(buffer, pagesize=A4, footer_text=footer_text)
    largura, altura = A4

    margem_esq = 2 * cm
    margem_dir = 2 * cm
    margem_inf = 2 * cm
    y = altura - 2 * cm

    # --- Banner no topo (proporcional, sem deformar) ---
    banner = Path(__file__).parent / "assets" / "banner.pdf.a4.png"
    largura_util = largura - margem_esq - margem_dir
    max_banner_h = 3.2 * cm  # ajuste fino: 2.8–3.6 cm

    if banner.exists():
        img = ImageReader(str(banner))
        iw, ih = img.getSize()

        # escala para caber na largura
        escala = largura_util / float(iw)
        w = largura_util
        h = ih * escala

        # se ficou alto demais, limita pela altura máxima (mantém proporção)
        if h > max_banner_h:
            escala = max_banner_h / float(ih)
            h = max_banner_h
            w = iw * escala

        # centraliza horizontalmente se sobrou espaço (quando limitou pela altura)
        x = margem_esq + (largura_util - w) / 2.0

        c.drawImage(
            img,
            x,
            y - h,
            width=w,
            height=h,
            preserveAspectRatio=True,
            mask="auto",
        )
        y -= h + 0.8 * cm
    else:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(margem_esq, y, f"Banner não encontrado: {banner}")
        y -= 0.8 * cm

    c.setFont("Helvetica", 10)
    for linha in wrap_text(pergunta):
        if y < (margem_inf + 1.6 * cm):  # reserva espaço pro rodapé
            c.showPage()
            y = altura - 2 * cm
            c.setFont("Helvetica", 10)
        c.drawString(margem_esq, y, linha)
        y -= 0.45 * cm

    y -= 0.8 * cm
    if y < (margem_inf + 1.6 * cm):
        c.showPage()
        y = altura - 2 * cm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margem_esq, y, "Resposta do Sistema")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    for linha in wrap_text(resposta):
        if y < (margem_inf + 1.6 * cm):
            c.showPage()
            y = altura - 2 * cm
            c.setFont("Helvetica", 10)
        c.drawString(margem_esq, y, linha)
        y -= 0.45 * cm

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# =========================
# Execução
# =========================
# O envio é feito pelo botão dentro do componente HTML acima
if enviar and prompt:
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

            # --- Sugestão de esboço (quando fizer sentido) ---
            if auto_sketch:
                with st.spinner("Checando se um esboço ajudaria…"):
                    d = decidir_esboco(prompt, final_text)
                if d.get("need_sketch"):
                    st.markdown("### ✍️ Sugestão de esboço")
                    if d.get("reason"):
                        st.info(d["reason"])

                    # BÔNUS ELEGANTE: mostra o prompt com wrap + evita variável não definida
                    sketch = (d.get("sketch_prompt") or "").strip()
                    if sketch:
                        st.info("✅ O sistema sugere que um esboço ajudaria. Copie o prompt abaixo.")
                        st.text_area(
                            "Prompt do esboço (PT-BR) — copie e cole no gerador de imagens",
                            value=sketch,
                            height=200,
                        )
                        
                        # Botões de download e copiar lado a lado
                        col_download, col_copy_sketch = st.columns([1, 1])
                        
                        with col_download:
                            st.download_button(
                                "💾 Baixar prompt (.txt)",
                                data=sketch.encode("utf-8"),
                                file_name="prompt_esboco.txt",
                                mime="text/plain; charset=utf-8",
                                use_container_width=True,
                            )
                        
                        with col_copy_sketch:
                            # Botão de copiar o prompt do esboço (azul marinho)
                            copy_sketch_html = f"""
                            <button onclick="copySketchPrompt()" 
                                    style="width:100%; padding:0.5rem 1rem; background-color:#1e3a8a; 
                                           color:white; border:1px solid #2563eb; border-radius:0.5rem; 
                                           cursor:pointer; font-size:0.9rem; font-weight:500;">
                                📋 Copiar prompt
                            </button>
                            <textarea id="sketchTextToCopy" style="position:absolute; left:-9999px;">{sketch}</textarea>
                            <script>
                            function copySketchPrompt() {{
                                const text = document.getElementById('sketchTextToCopy').value;
                                navigator.clipboard.writeText(text).then(function() {{
                                    const btn = event.target;
                                    const original = btn.innerHTML;
                                    btn.innerHTML = '✅ Copiado!';
                                    btn.style.backgroundColor = '#0e7c0e';
                                    setTimeout(function() {{
                                        btn.innerHTML = original;
                                        btn.style.backgroundColor = '#1e3a8a';
                                    }}, 2000);
                                }}, function(err) {{
                                    alert('Erro ao copiar: ' + err);
                                }});
                            }}
                            </script>
                            """
                            components.html(copy_sketch_html, height=50)
                    else:
                        st.caption("O decisor marcou que um esboço ajudaria, mas não gerou um prompt. Tente novamente ou reformule o caso.")

                    st.caption(
                        "Dica: cole esse prompt em um gerador de imagens (NanoBanana, Leonardo, Stable Diffusion, etc.). "
                        "Se quiser, eu também posso adaptar o prompt para a ferramenta que você for usar."
                    )

            # Guarda para exportação em PDF
            st.session_state["ultima_pergunta"] = prompt
            st.session_state["ultima_resposta"] = final_text
            st.session_state["ultimo_modelo"] = model_name
            st.session_state["ultimo_modo"] = mode

            # Botão de exportação (fica logo após a resposta)
            st.markdown("<h4 style='font-size:1.1rem; margin-top:1.5rem;'>📄 Exportar resposta</h4>", unsafe_allow_html=True)
            
            # Dois botões lado a lado
            col_pdf, col_copy = st.columns([1, 1])
            
            with col_pdf:
                if not _PDF_OK:
                    st.warning("Exportação PDF indisponível: instale `reportlab` no requirements.txt.")
                else:
                    pdf_bytes = gerar_pdf_a4(prompt, final_text)
                    st.download_button(
                        "📥 Gerar PDF A4",
                        data=pdf_bytes,
                        file_name="ensina_feridas_resposta.pdf",
                        mime="application/pdf",
                        key="download_pdf_a4",
                        use_container_width=True,
                    )
            
            with col_copy:
                # Prepara texto completo para copiar
                texto_completo = f"PERGUNTA:\n{prompt}\n\n{'='*50}\n\nRESPOSTA:\n{final_text}"
                
                # Botão de copiar com JavaScript (azul marinho)
                copy_button_html = f"""
                <button onclick="copyToClipboard()" 
                        style="width:100%; padding:0.5rem 1rem; background-color:#1e3a8a; 
                               color:white; border:1px solid #2563eb; border-radius:0.5rem; 
                               cursor:pointer; font-size:0.9rem; font-weight:500;">
                    📋 Copiar texto
                </button>
                <textarea id="textToCopy" style="position:absolute; left:-9999px;">{texto_completo}</textarea>
                <script>
                function copyToClipboard() {{
                    const text = document.getElementById('textToCopy').value;
                    navigator.clipboard.writeText(text).then(function() {{
                        const btn = event.target;
                        const original = btn.innerHTML;
                        btn.innerHTML = '✅ Copiado!';
                        btn.style.backgroundColor = '#0e7c0e';
                        setTimeout(function() {{
                            btn.innerHTML = original;
                            btn.style.backgroundColor = '#1e3a8a';
                        }}, 2000);
                    }}, function(err) {{
                        alert('Erro ao copiar: ' + err);
                    }});
                }}
                </script>
                """
                components.html(copy_button_html, height=50)

        except Exception as e:
            st.error("Erro ao chamar o Gemini:")
            st.exception(e)

st.divider()

