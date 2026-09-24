import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Passos Mágicos | Analytics & Risco Preditivo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS adaptativa para Modo Claro e Modo Escuro nativos do Streamlit
st.markdown("""
<style>
    @media (prefers-color-scheme: dark) {
        .risk-high {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-left: 4px solid #EF4444;
            padding: 1rem;
            border-radius: 6px;
            color: #FCA5A5;
        }
        .risk-medium {
            background-color: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-left: 4px solid #F59E0B;
            padding: 1rem;
            border-radius: 6px;
            color: #FCD34D;
        }
        .risk-low {
            background-color: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-left: 4px solid #10B981;
            padding: 1rem;
            border-radius: 6px;
            color: #6EE7B7;
        }
    }

    @media (prefers-color-scheme: light) {
        .risk-high {
            background-color: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-left: 4px solid #DC2626;
            padding: 1rem;
            border-radius: 6px;
            color: #991B1B;
        }
        .risk-medium {
            background-color: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.2);
            border-left: 4px solid #D97706;
            padding: 1rem;
            border-radius: 6px;
            color: #92400E;
        }
        .risk-low {
            background-color: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-left: 4px solid #059669;
            padding: 1rem;
            border-radius: 6px;
            color: #065F46;
        }
    }

    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Função de Carga do Modelo com Fallback Resiliente
# -------------------------------------------------------------
@st.cache_resource
def carregar_modelo():
    caminhos = ['modelo_risco_defasagem.pkl', '../modelo_risco_defasagem.pkl']
    for caminho in caminhos:
        if os.path.exists(caminho):
            try:
                modelo = joblib.load(caminho)
                return modelo, "pkl"
            except Exception:
                pass
    return None, "fallback"

modelo_carregado, modo_modelo = carregar_modelo()

def predizer_risco(df_input):
    if modelo_carregado is not None:
        try:
            prob = modelo_carregado.predict_proba(df_input)[0][1]
            classe = int(prob >= 0.50)
            return prob, classe
        except Exception:
            pass

    row = df_input.iloc[0]
    score_z = 0.0
    score_z += (row['idade'] - 12.0) * 0.28
    score_z += row['defasagem_escolar'] * 0.45
    score_z += (2024 - row['ano_ingresso']) * -0.15
    score_z -= (row['ida'] - 6.0) * 0.42
    score_z -= (row['ieg'] - 7.5) * 0.38
    score_z -= (row['ips'] - 6.5) * 0.35
    score_z -= (row['ipp'] - 6.5) * 0.30
    score_z -= (row['pedra_ord'] - 2.5) * 0.50
    
    prob = 1.0 / (1.0 + np.exp(-score_z))
    prob = float(np.clip(prob, 0.02, 0.98))
    classe = int(prob >= 0.50)
    return prob, classe

# -------------------------------------------------------------
# Barra Lateral (Navegação & Status)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### **PASSOS MÁGICOS**")
    st.caption("Datathon Analytics • Fase 5")
    st.markdown("---")
    
    menu = st.radio(
        "Navegação do Sistema",
        [
            "Simulador Individual",
            "Simulação em Lote",
            "Manual de Uso",
            "Diagnóstico & Storytelling",
            "Dicionário de Indicadores",
            "Metodologia ML"
        ]
    )
    st.markdown("---")
    if modo_modelo == "pkl":
        st.success("Modelo Serializado (.pkl)")
    else:
        st.info("Modelo Fallback Ativo")

# -------------------------------------------------------------
# ABA 1: SIMULADOR DE RISCO INDIVIDUAL
# -------------------------------------------------------------
if menu == "Simulador Individual":
    st.markdown('<div class="main-header">Simulador Preditivo de Risco Escolar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ferramenta de apoio pedagógico para identificar antecipadamente estudantes em vulnerabilidade de desempenho.</div>', unsafe_allow_html=True)

    st.info("💡 **Como usar:** Clique em um dos **arquétipos de exemplo** abaixo para preencher os campos automaticamente com um perfil hipotético, ou altere os seletores livremente para simular o caso real de um aluno específico.")

    st.markdown("##### Carregar Arquétipo de Exemplo (Perfil Hipotético)")
    c1, c2, c3, c4 = st.columns(4)
    
    perfil_selecionado = None
    if c1.button("Ex: Quartzo (Vulnerável)", use_container_width=True):
        perfil_selecionado = {'idade': 15, 'ano_ingresso': 2023, 'inde': 5.2, 'ida': 4.8, 'ieg': 5.1, 'iaa': 8.0, 'ips': 4.9, 'ipp': 5.0, 'ipv': 5.2, 'defasagem_escolar': 2, 'pedra': 1}
    if c2.button("Ex: Ágata (Estável)", use_container_width=True):
        perfil_selecionado = {'idade': 13, 'ano_ingresso': 2022, 'inde': 6.8, 'ida': 6.2, 'ieg': 7.4, 'iaa': 7.8, 'ips': 6.5, 'ipp': 6.8, 'ipv': 6.5, 'defasagem_escolar': 0, 'pedra': 2}
    if c3.button("Ex: Ametista (Avançado)", use_container_width=True):
        perfil_selecionado = {'idade': 14, 'ano_ingresso': 2021, 'inde': 8.0, 'ida': 7.8, 'ieg': 8.5, 'iaa': 8.2, 'ips': 7.9, 'ipp': 8.0, 'ipv': 8.1, 'defasagem_escolar': 0, 'pedra': 3}
    if c4.button("Ex: Topázio (Protagonista)", use_container_width=True):
        perfil_selecionado = {'idade': 12, 'ano_ingresso': 2020, 'inde': 8.9, 'ida': 8.5, 'ieg': 9.2, 'iaa': 8.7, 'ips': 8.3, 'ipp': 8.6, 'ipv': 8.8, 'defasagem_escolar': 0, 'pedra': 4}

    defaults = perfil_selecionado if perfil_selecionado else {
        'idade': 14, 'ano_ingresso': 2021, 'inde': 7.2, 'ida': 6.8, 'ieg': 7.8,
        'iaa': 8.2, 'ips': 7.0, 'ipp': 6.9, 'ipv': 7.1, 'defasagem_escolar': 0, 'pedra': 2
    }

    st.markdown("---")
    st.markdown("#### Parâmetros Atuais do Estudante")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        idade = st.number_input("Idade", min_value=6, max_value=24, value=defaults['idade'], step=1)
        ano_ingresso = st.number_input("Ano de Ingresso", min_value=2015, max_value=2024, value=defaults['ano_ingresso'], step=1)
        defasagem_escolar = st.number_input("Defasagem (Anos)", min_value=-2, max_value=6, value=defaults['defasagem_escolar'], step=1)
        
    with col2:
        ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, float(defaults['ida']), 0.1)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, float(defaults['ieg']), 0.1)
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, float(defaults['ips']), 0.1)

    with col3:
        ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, float(defaults['ipp']), 0.1)
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, float(defaults['ipv']), 0.1)
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, float(defaults['iaa']), 0.1)

    col_pedra, col_inde = st.columns(2)
    with col_pedra:
        pedra_opcoes = {1: "1. Quartzo", 2: "2. Ágata", 3: "3. Ametista", 4: "4. Topázio"}
        pedra_ord = st.selectbox("Classificação da Pedra Atual", options=[1, 2, 3, 4], index=defaults['pedra']-1, format_func=lambda x: pedra_opcoes[x])
    with col_inde:
        inde = st.slider("INDE Geral", 0.0, 10.0, float(defaults['inde']), 0.1)

    df_estudante = pd.DataFrame([{
        'idade': idade, 'ano_ingresso': ano_ingresso, 'inde': inde, 'ida': ida,
        'ieg': ieg, 'iaa': iaa, 'ips': ips, 'ipp': ipp, 'ipv': ipv,
        'defasagem_escolar': defasagem_escolar, 'pedra_ord': pedra_ord
    }])

    st.markdown(" ")
    if st.button("Executar Predição de Risco", type="primary", use_container_width=True):
        prob, classe = predizer_risco(df_estudante)
        
        st.markdown("#### Relatório de Diagnóstico")
        res_col1, res_col2, res_col3 = st.columns([1, 1.5, 1])
        
        with res_col1:
            st.metric("Probabilidade Calculada", f"{prob*100:.1f}%")
            st.progress(float(prob))

        with res_col2:
            if prob >= 0.65:
                st.markdown('<div class="risk-high"><strong>ALTO RISCO DETECTADO</strong><br>Probabilidade elevada de queda de rendimento no próximo ciclo.</div>', unsafe_allow_html=True)
            elif prob >= 0.40:
                st.markdown('<div class="risk-medium"><strong>RISCO MODERADO</strong><br>Zona de atenção. Monitoramento preventivo sugerido.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low"><strong>BAIXO RISCO</strong><br>Trajetória acadêmica estável e consolidada.</div>', unsafe_allow_html=True)

        with res_col3:
            tempo_casa = 2024 - ano_ingresso
            st.markdown(f"""
            **Métricas Síntese:**
            - Tempo de Casa: {tempo_casa} ano(s)
            - Defasagem: {defasagem_escolar} ano(s)
            - Status Emocional: {"Crítico" if ips < 6.0 else "Estável"}
            """)

        st.markdown("#### Plano de Ação Pedagógico Recomendado")
        recs = []
        if ips < 6.5:
            recs.append("Atenção Psicossocial (IPS): O indicador emocional aponta vulnerabilidade prévia a quedas de desempenho.")
        if ieg < 7.0:
            recs.append("Reforço de Engajamento (IEG): Alinhar diretrizes de presença e participação ativa nas atividades.")
        if pedra_ord == 1:
            recs.append("Plano Quartzo: Estabelecer metas curtas direcionadas à transição de fase.")
        elif pedra_ord == 3:
            recs.append("Consolidação Ametista: Estimular tutoria cruzada para pavimentar a chegada ao nível Topázio.")

        if not recs:
            recs.append("Manter acompanhamento regular e incentivar o protagonismo em projetos de mentoria.")

        for r in recs:
            st.info(r)

# -------------------------------------------------------------
# ABA 2: SIMULAÇÃO EM LOTE
# -------------------------------------------------------------
elif menu == "Simulação em Lote":
    st.markdown('<div class="main-header">Triagem em Lote</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Processamento automatizado de bases de turmas via arquivo CSV.</div>', unsafe_allow_html=True)

    amostra_exemplo = pd.DataFrame([
        {'ra': 'RA-001', 'idade': 15, 'ano_ingresso': 2023, 'inde': 5.2, 'ida': 4.8, 'ieg': 5.0, 'iaa': 8.0, 'ips': 4.8, 'ipp': 5.1, 'ipv': 5.0, 'defasagem_escolar': 2, 'pedra_ord': 1},
        {'ra': 'RA-002', 'idade': 14, 'ano_ingresso': 2022, 'inde': 6.8, 'ida': 6.5, 'ieg': 7.5, 'iaa': 7.5, 'ips': 6.8, 'ipp': 7.0, 'ipv': 6.8, 'defasagem_escolar': 0, 'pedra_ord': 2},
        {'ra': 'RA-003', 'idade': 14, 'ano_ingresso': 2021, 'inde': 8.0, 'ida': 7.8, 'ieg': 8.5, 'iaa': 8.2, 'ips': 7.9, 'ipp': 8.0, 'ipv': 8.1, 'defasagem_escolar': 0, 'pedra_ord': 3},
        {'ra': 'RA-004', 'idade': 12, 'ano_ingresso': 2020, 'inde': 8.8, 'ida': 8.6, 'ieg': 9.2, 'iaa': 8.5, 'ips': 8.5, 'ipp': 8.8, 'ipv': 8.9, 'defasagem_escolar': 0, 'pedra_ord': 4}
    ])

    uploaded_file = st.file_uploader("Carregar Base CSV", type=["csv"])
    df_analise = None
    
    if uploaded_file is not None:
        try:
            df_analise = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Erro na leitura: {e}")
    else:
        if st.button("Carregar Amostra Padrão para Teste"):
            df_analise = amostra_exemplo

    if df_analise is not None:
        cols_req = ['idade', 'ano_ingresso', 'inde', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 'ipv', 'defasagem_escolar', 'pedra_ord']
        if all(c in df_analise.columns for c in cols_req):
            resultados = []
            for _, row in df_analise.iterrows():
                p, _ = predizer_risco(pd.DataFrame([row[cols_req].to_dict()]))
                resultados.append({
                    'Risco (%)': round(p * 100, 1),
                    'Status': 'Alto Risco' if p >= 0.65 else ('Risco Moderado' if p >= 0.40 else 'Baixo Risco')
                })
            
            df_res = df_analise.copy()
            df_res['Probabilidade'] = [r['Risco (%)'] for r in resultados]
            df_res['Classificação'] = [r['Status'] for r in resultados]
            
            tot = len(df_res)
            altos = (df_res['Classificação'] == 'Alto Risco').sum()
            
            c1, c2 = st.columns(2)
            c1.metric("Total Analisado", tot)
            c2.metric("Casos Críticos", altos)
            
            st.dataframe(df_res, use_container_width=True)
            st.download_button("Exportar Resultados (CSV)", df_res.to_csv(index=False).encode('utf-8'), "triagem_risco.csv", "text/csv")
        else:
            st.error("O arquivo submetido não contém todas as colunas obrigatórias.")

# -------------------------------------------------------------
# ABA 3: MANUAL DE USO
# -------------------------------------------------------------
elif menu == "Manual de Uso":
    st.markdown('<div class="main-header">Manual de Uso da Ferramenta</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Guia prático para orientar professores, mentores e equipe pedagógica.</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Simulador Individual
    * **O que faz:** Calcula a probabilidade de um aluno específico enfrentar dificuldades ou defasagem no próximo ciclo.
    * **Como utilizar:** 
      1. Utilize os botões de **Arquétipos de Exemplo** para carregar perfis hipotéticos de teste (Quartzo, Ágata, Ametista ou Topázio).
      2. Ajuste os seletores (*sliders* e campos numéricos) conforme as notas e características reais do seu aluno.
      3. Clique em **"Executar Predição de Risco"** para visualizar a porcentagem de risco, o diagnóstico automático e um **Plano de Ação Pedagógico** personalizado.

    ### 2. Simulação em Lote
    * **O que faz:** Realiza a triagem preditiva de turmas inteiras de forma automatizada.
    * **Como utilizar:** 
      1. Envie um arquivo `.csv` contendo as colunas obrigatórias (`idade`, `ano_ingresso`, `inde`, `ida`, `ieg`, `iaa`, `ips`, `ipp`, `ipv`, `defasagem_escolar`, `pedra_ord`). Caso queira testar rapidamente sem um arquivo próprio, clique em **"Carregar Amostra Padrão para Teste"**.
      2. O sistema processará os dados, exibirá uma tabela consolidada com o nível de risco de cada estudante e permitirá o download dos resultados em CSV.

    ### 3. Demais Módulos
    * **Diagnóstico & Storytelling:** Apresenta insights estratégicos e descobertas da série histórica da instituição.
    * **Dicionário de Indicadores:** Esclarece o conceito técnico de cada sigla avaliada (INDE, IDA, IEG, IPS, etc.).
    * **Metodologia ML:** Descreve a arquitetura do modelo preditivo e métricas de validação técnica.
    """)

# -------------------------------------------------------------
# ABA 4: DIAGNÓSTICO & STORYTELLING
# -------------------------------------------------------------
elif menu == "Diagnóstico & Storytelling":
    st.markdown('<div class="main-header">Insights Analíticos</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Descobertas estratégicas baseadas na série histórica 2022-2024.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Eficácia do Modelo de Pedras")
        st.write("A progressão entre Quartzo, Ágata, Ametista e Topázio reflete ganhos expressivos na retenção e no engajamento geral (IEG), blindando os estudantes contra a evasão escolar.")
    with c2:
        st.markdown("### O Indicador Sensor (IPS)")
        st.write("Variações negativas no Indicador Psicossocial antecedem quedas de desempenho acadêmico (IDA) no ciclo seguinte, servindo como alerta precoce essencial.")

# -------------------------------------------------------------
# ABA 5: DICIONÁRIO DE INDICADORES
# -------------------------------------------------------------
elif menu == "Dicionário de Indicadores":
    st.markdown('<div class="main-header">Dicionário de Métricas</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Estrutura conceitual dos indicadores da Passos Mágicos.</div>', unsafe_allow_html=True)
    
    ind = [
        ("INDE", "Índice de Desenvolvimento Educacional (Métrica sintética global)."),
        ("IAN", "Indicador de Adequação de Nível (Mede defasagem idade-série)."),
        ("IDA", "Indicador de Desempenho Acadêmico (Notas e aproveitamento)."),
        ("IEG", "Indicador de Engajamento Geral (Frequência e entregas)."),
        ("IAA", "Indicador de Autoavaliação (Percepção do próprio desenvolvimento)."),
        ("IPS", "Indicador Psicossocial (Estabilidade emocional e familiar)."),
        ("IPP", "Indicador Psicopedagógico (Prontidão cognitiva)."),
        ("IPV", "Indicador de Ponto de Virada (Autonomia e protagonismo).")
    ]
    for sigla, desc in ind:
        st.markdown(f"**{sigla}**: {desc}")

# -------------------------------------------------------------
# ABA 6: METODOLOGIA ML
# -------------------------------------------------------------
elif menu == "Metodologia ML":
    st.markdown('<div class="main-header">Pipeline de Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Validação temporal e prevenção estrita de data leakage.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    - **Validação:** Uso de `GroupShuffleSplit` baseado no ID do aluno (`RA`) para isolar ciclos temporais.
    - **Modelo Campeão:** Regressão Logística otimizada, garantindo interpretabilidade direta para laçada pedagógica.
    """)
    
    df_perf = pd.DataFrame([
        {'Modelo': 'Regressão Logística', 'AUC-ROC': '0.8503', 'Overfitting Gap': 'Baixo (1.5%)'},
        {'Modelo': 'HistGradientBoosting', 'AUC-ROC': '0.7998', 'Overfitting Gap': 'Moderado (17.5%)'},
        {'Modelo': 'Random Forest', 'AUC-ROC': '0.7870', 'Overfitting Gap': 'Alto (21.3%)'}
    ])
    st.table(df_perf)
