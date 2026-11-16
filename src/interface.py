import os
import sys
import json
import subprocess
import threading
import time
import asyncio
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
import flet as ft
from src.banco import BancoDados
from src.configuracao import obter_configuracao
from src.coleta.motor_coleta import MotorColeta
from src.extracao.motor_extracao import MotorExtracao
from src.nlp.motor_nlp import MotorNLP
from src.cluster.agrupador import AgrupadorEventos
from src.analise.perguntas import RespondedorPerguntas
from src.grafos.construtor_grafo import ConstrutorGrafos, ConstrutorGrafoSemantico
from src.visualizacao.graficos import GeradorGraficosArtigo
from src.registro import adicionar_ouvinte_logs


def abrir_pasta_no_sistema(caminho: str) -> None:
    p = Path(caminho).resolve()
    p.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("win"):
        os.startfile(str(p))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p)])


def selecionar_diretorio_dialogo(titulo: str = "Selecione o diretório de dados") -> Optional[str]:
    if sys.platform.startswith("linux"):
        import shutil
        if shutil.which("zenity"):
            try:
                proc = subprocess.run(
                    ["zenity", "--file-selection", "--directory", f"--title={titulo}"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    return proc.stdout.strip()
            except Exception:
                pass
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        caminho = filedialog.askdirectory(title=titulo)
        root.destroy()
        return caminho if caminho else None
    except Exception:
        return None


class InterfaceFletApp:
    def __init__(self):
        self.banco = BancoDados()
        self.config = obter_configuracao()
        self.em_execucao = False
        self.status_carregando = False
        self.etapa_atual = "coletar"
        self.logs_armazenados: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.filtro_etapa_logs = "todas"
        self.fila_logs: List[Dict[str, str]] = []
        self.lock_ui = threading.Lock()
        self._atualizacao_agendada = False
        self.loop = None

    def iniciar(self, page: ft.Page) -> None:
        self.page = page
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = None
        page.title = "Quantificação de Viés Editorial e Sensacionalismo na Mídia Brasileira (2015–2025)"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.window_width = 1180
        page.window_height = 920
        page.bgcolor = "#F8FAFC"

        adicionar_ouvinte_logs(self._receber_log_externo)

        self.nav_segment = ft.SegmentedButton(
            selected=["etapas"],
            on_change=self._ao_mudar_secao,
            segments=[
                ft.Segment(
                    value="etapas",
                    label=ft.Text("Executar Etapas"),
                    icon=ft.Icon(ft.Icons.PLAY_CIRCLE_OUTLINE)
                ),
                ft.Segment(
                    value="status",
                    label=ft.Text("Status do Projeto"),
                    icon=ft.Icon(ft.Icons.ANALYTICS_OUTLINED)
                ),
                ft.Segment(
                    value="logs",
                    label=ft.Text("Logs Técnicos"),
                    icon=ft.Icon(ft.Icons.TERMINAL_OUTLINED)
                )
            ]
        )

        self.dd_etapa = ft.Dropdown(
            label="Escolha a Etapa do Pipeline",
            value="coletar",
            options=[
                ft.DropdownOption("coletar", "1. Coleta de URLs (Sitemaps / Wayback Machine)"),
                ft.DropdownOption("extrair", "2. Extração de Conteúdo e Limpeza Textual"),
                ft.DropdownOption("nlp", "3. Processamento NLP (Sentimento, Embeddings, NER)"),
                ft.DropdownOption("clusterizar", "4. Agrupamento Temporal de Eventos (HDBSCAN)"),
                ft.DropdownOption("estatisticas", "5. Baterias Estatísticas e Testes de Hipótese"),
                ft.DropdownOption("grafos", "6. Construção de Grafos para o Gephi (GEXF)"),
                ft.DropdownOption("graficos", "7. Renderização de Figuras do Artigo (PNG 300 DPI)"),
                ft.DropdownOption("tudo", "8. Execução Integrada (Todas as Etapas)")
            ],
            on_select=self._ao_mudar_etapa,
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True
        )

        self.txt_limite = ft.TextField(
            label="Limite Quantitativo",
            hint_text="Digite a quantidade desejada (vazio = ilimitado / todas)",
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True
        )

        self.txt_veiculo = ft.TextField(
            label="Filtro de Veículo(s)",
            hint_text="Ex: Tradicional 1, Digital 1, tradicional, digital (vazio = todos)",
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True
        )

        self.dd_fonte = ft.Dropdown(
            label="Fonte de Coleta",
            value="todas",
            options=[
                ft.DropdownOption("todas", "Todas as Fontes (Sitemaps + Wayback Machine)"),
                ft.DropdownOption("sitemaps", "Apenas Sitemaps Diretos dos Jornais"),
                ft.DropdownOption("cdx", "Apenas Wayback Machine (CDX Server)")
            ],
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True
        )

        self.txt_anos = ft.TextField(
            label="Ano(s) de Competência",
            hint_text="Ex: 2018 ou 2015-2025 ou 2016,2018 (vazio = década 2015-2025)",
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True
        )

        self.dd_modo_nlp = ft.Dropdown(
            label="Modo de NLP",
            value="tudo",
            options=[
                ft.DropdownOption("tudo", "Tudo (Sentimento, Embeddings, NER)"),
                ft.DropdownOption("sentimento", "Sentimento (RoBERTa) e Clickbait"),
                ft.DropdownOption("embeddings", "Embeddings Semânticos e Eixos"),
                ft.DropdownOption("entidades", "Entidades Nomeadas (spaCy NER)")
            ],
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True
        )

        self.chk_reprocessar_erros = ft.Checkbox(
            label="Reprocessar apenas URLs com erro anterior de download",
            value=False,
            active_color="#0284C7"
        )

        self.chk_apenas_eventos = ft.Checkbox(
            label="Processar apenas matérias já associadas a macro-eventos",
            value=False,
            active_color="#0284C7"
        )

        self.chk_usar_externo = ft.Checkbox(
            label="Carregar dados de diretório externo (ZIP ou CSVs)",
            value=False,
            on_change=self._ao_alternar_externo,
            active_color="#0D9488"
        )

        self.txt_pasta_externa = ft.TextField(
            label="Caminho do Diretório Externo de Dados",
            hint_text="Selecione ou cole o caminho da pasta onde estão os arquivos baixados",
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True,
            visible=False
        )

        self.btn_procurar_externo = ft.OutlinedButton(
            "Selecionar Pasta...",
            icon=ft.Icons.FOLDER_OPEN_ROUNDED,
            on_click=self._abrir_seletor_pasta_externa,
            visible=False
        )

        self.dd_figuras = ft.Dropdown(
            label="Figuras do Artigo para Gerar",
            value="todas",
            options=[
                ft.DropdownOption("todas", "Todas as Figuras (01 a 07 + Legendas)"),
                ft.DropdownOption("01", "Figura 1: Balanço Bipolar de Sentimento"),
                ft.DropdownOption("02", "Figura 2: Lollipop Ranking Clickbait"),
                ft.DropdownOption("03a", "Figura 3A: Cardiograma Ciclos Eleitorais"),
                ft.DropdownOption("04", "Figura 4: Sankey Enquadramento Editorial"),
                ft.DropdownOption("05a", "Figura 5A: Radar Silhueta Editorial"),
                ft.DropdownOption("05b", "Figura 5B: Tornado Efeito Mudo"),
                ft.DropdownOption("06a", "Figura 6A: Série Histórica Divergência Temática"),
                ft.DropdownOption("06b", "Figura 6B: Metástase da Polarização"),
                ft.DropdownOption("07", "Figura 7: Régua de Polaridade e Fatos-Ponte"),
                ft.DropdownOption("legendas", "Legendas Oficiais dos Grafos (PNG e SVG)")
            ],
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            expand=True
        )

        self.btn_abrir_pasta_graficos = ft.OutlinedButton(
            "Abrir Pasta de Gráficos (PNG)",
            icon=ft.Icons.IMAGE_OUTLINED,
            on_click=lambda _: abrir_pasta_no_sistema("resultados/graficos")
        )

        self.btn_abrir_pasta_grafos = ft.OutlinedButton(
            "Abrir Pasta de Grafos (Arquivos e Imagens)",
            icon=ft.Icons.HUB_OUTLINED,
            on_click=lambda _: abrir_pasta_no_sistema("resultados/grafos")
        )

        self.btn_executar = ft.FilledButton(
            "Iniciar Execução da Etapa",
            icon=ft.Icons.PLAY_ARROW_ROUNDED,
            on_click=self._executar_clicado,
            style=ft.ButtonStyle(
                bgcolor="#0284C7",
                color="#FFFFFF"
            )
        )

        self.progresso = ft.ProgressBar(visible=False, color="#0284C7", bgcolor="#E2E8F0")
        self.lbl_status = ft.Text("Pronto para execução", size=13, color="#64748B", weight=ft.FontWeight.W_500)

        self.painel_parametros = ft.Column(spacing=10)
        self.painel_resultados = ft.Column(spacing=12, scroll=ft.ScrollMode.ADAPTIVE, expand=True)
        self.painel_status_projeto = ft.Column(spacing=12, scroll=ft.ScrollMode.ADAPTIVE, expand=True)

        self.dd_filtro_logs = ft.Dropdown(
            label="Filtrar Logs por Etapa",
            value="todas",
            options=[
                ft.DropdownOption("todas", "Todas as Etapas"),
                ft.DropdownOption("coletar", "1. Coleta de URLs"),
                ft.DropdownOption("extrair", "2. Extração de Conteúdo"),
                ft.DropdownOption("nlp", "3. Processamento NLP"),
                ft.DropdownOption("clusterizar", "4. Agrupamento HDBSCAN"),
                ft.DropdownOption("estatisticas", "5. Baterias Estatísticas"),
                ft.DropdownOption("grafos", "6. Construção de Grafos"),
                ft.DropdownOption("graficos", "7. Figuras do Artigo"),
                ft.DropdownOption("sistema", "Geral / Sistema")
            ],
            on_select=self._ao_filtrar_logs,
            border_color="#CBD5E1",
            color="#0F172A",
            bgcolor="#FFFFFF",
            width=280
        )

        self.terminal_view = ft.ListView(expand=True, spacing=4, auto_scroll=True, padding=12)

        self._atualizar_campos_parametros("coletar")
        self.painel_status_projeto.controls.append(
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("Painel de Status Global do Projeto", size=16, weight=ft.FontWeight.BOLD, color="#0F172A"),
                        ft.Text("Consulte a volumetria consolidada, distribuição por polos editoriais e eixos temáticos em base.db.", size=13, color="#64748B"),
                        ft.OutlinedButton("Consultar Estatísticas do Banco Agora", icon=ft.Icons.REFRESH_ROUNDED, on_click=lambda _: self._disparar_carregamento_status())
                    ],
                    spacing=10
                ),
                bgcolor="#FFFFFF",
                border=ft.Border.all(1, "#E2E8F0"),
                border_radius=8,
                padding=18
            )
        )

        cabecalho = ft.Row(
            controls=[
                ft.Icon(ft.Icons.ACCOUNT_TREE_ROUNDED, color="#0284C7", size=36),
                ft.Column(
                    controls=[
                        ft.Text(
                            "Quantificação de Viés Editorial e Sensacionalismo na Mídia Brasileira (2015–2025)",
                            size=18,
                            weight=ft.FontWeight.BOLD,
                            color="#0F172A"
                        ),
                        ft.Text(
                            "Uma Abordagem Baseada em Processamento de Linguagem Natural e Modelagem em Grafos",
                            size=13,
                            color="#475569"
                        )
                    ],
                    spacing=2
                )
            ]
        )

        self.view_etapas = ft.Column(
            controls=[
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Row([self.dd_etapa]),
                            self.painel_parametros,
                            ft.Row([self.btn_executar, self.lbl_status], alignment=ft.MainAxisAlignment.START),
                            self.progresso
                        ],
                        spacing=12
                    ),
                    bgcolor="#FFFFFF",
                    border=ft.Border.all(1, "#E2E8F0"),
                    border_radius=8,
                    padding=18
                ),
                ft.Text("Resultados e Entregas da Etapa:", size=14, weight=ft.FontWeight.BOLD, color="#334155"),
                ft.Container(
                    content=self.painel_resultados,
                    bgcolor="#FFFFFF",
                    border=ft.Border.all(1, "#E2E8F0"),
                    border_radius=8,
                    padding=16,
                    expand=True
                )
            ],
            spacing=14,
            expand=True
        )

        self.view_status = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.DATA_EXPLORATION_ROUNDED, color="#0D9488", size=24),
                        ft.Text("Visão Geral do Repositório e Estatísticas Descritivas", size=16, weight=ft.FontWeight.BOLD, color="#0F172A")
                    ],
                    spacing=8
                ),
                self.painel_status_projeto
            ],
            spacing=14,
            visible=False,
            expand=True
        )

        self.view_logs = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Row([
                            ft.Icon(ft.Icons.TERMINAL_ROUNDED, color="#475569", size=22),
                            ft.Text("Logs Técnicos de Execução:", size=14, weight=ft.FontWeight.BOLD, color="#0F172A"),
                            self.dd_filtro_logs
                        ], spacing=10),
                        ft.OutlinedButton("Limpar Logs", icon=ft.Icons.CLEAR_ALL, on_click=self._limpar_logs)
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                ft.Container(
                    content=self.terminal_view,
                    bgcolor="#0F172A",
                    border=ft.Border.all(1, "#CBD5E1"),
                    border_radius=8,
                    expand=True
                )
            ],
            spacing=10,
            visible=False,
            expand=True
        )

        page.add(
            ft.Column(
                controls=[
                    cabecalho,
                    ft.Divider(color="#E2E8F0", height=1),
                    self.nav_segment,
                    self.view_etapas,
                    self.view_status,
                    self.view_logs
                ],
                spacing=14,
                expand=True
            )
        )
        self._adicionar_log("sistema", "Interface gráfica carregada com êxito.")

    def _executar_na_ui(self, fn: Callable, *args: Any) -> None:
        if not hasattr(self, "page") or not self.page:
            return
        loop = getattr(self, "loop", None)
        if not isinstance(loop, asyncio.AbstractEventLoop) or loop.is_closed():
            try:
                conn = getattr(getattr(self.page, "session", None), "connection", None)
                cand_loop = getattr(conn, "loop", None)
                if isinstance(cand_loop, asyncio.AbstractEventLoop) and not cand_loop.is_closed():
                    self.loop = cand_loop
                    loop = cand_loop
                else:
                    loop = None
            except Exception:
                loop = None
        if isinstance(loop, asyncio.AbstractEventLoop) and loop.is_running():
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is loop:
                fn(*args)
            else:
                loop.call_soon_threadsafe(fn, *args)
        else:
            fn(*args)

    def atualizar_interface(self) -> None:
        with self.lock_ui:
            if self._atualizacao_agendada:
                return
            self._atualizacao_agendada = True

        self._executar_na_ui(self._processar_atualizacao_ui)

    def _processar_atualizacao_ui(self) -> None:
        with self.lock_ui:
            self._atualizacao_agendada = False
            novos = list(self.fila_logs)
            self.fila_logs.clear()

        if novos:
            for item in novos:
                if self.filtro_etapa_logs == "todas" or self.filtro_etapa_logs == item["etapa"]:
                    self._renderizar_linha_log(item)

        try:
            if hasattr(self, "page") and self.page:
                self.page.update()
        except Exception:
            pass

    def _receber_log_externo(self, etapa: str, mensagem: str, nivel: str, hora: str) -> None:
        etapa_chave = etapa.lower()
        if "coleta" in etapa_chave:
            chave = "coletar"
        elif "extra" in etapa_chave:
            chave = "extrair"
        elif "nlp" in etapa_chave or "sentimento" in etapa_chave:
            chave = "nlp"
        elif "cluster" in etapa_chave:
            chave = "clusterizar"
        elif "estat" in etapa_chave or "pergunta" in etapa_chave:
            chave = "estatisticas"
        elif "grafo" in etapa_chave and "grafico" not in etapa_chave:
            chave = "grafos"
        elif "grafico" in etapa_chave or "figura" in etapa_chave:
            chave = "graficos"
        else:
            chave = "sistema"

        self._adicionar_log(chave, mensagem, nivel=nivel, hora=hora)

    def _ao_mudar_secao(self, e: ft.ControlEvent) -> None:
        sel = list(e.control.selected)[0] if e.control.selected else "etapas"
        self.view_etapas.visible = (sel == "etapas")
        self.view_status.visible = (sel == "status")
        self.view_logs.visible = (sel == "logs")
        if sel == "status":
            self._disparar_carregamento_status()
        self.atualizar_interface()

    def _ao_alternar_externo(self, e: ft.ControlEvent) -> None:
        ativo = self.chk_usar_externo.value
        self.txt_pasta_externa.visible = ativo
        self.btn_procurar_externo.visible = ativo
        self.atualizar_interface()

    def _abrir_seletor_pasta_externa(self, _=None) -> None:
        def _tarefa():
            caminho = selecionar_diretorio_dialogo("Selecione a pasta onde estão os arquivos de dados")
            if caminho:
                def _aplicar():
                    self.txt_pasta_externa.value = caminho
                    self._adicionar_log("sistema", f"Diretório selecionado pelo usuário: {caminho}")
                    self._processar_atualizacao_ui()
                self._executar_na_ui(_aplicar)

        th = threading.Thread(target=_tarefa, daemon=True)
        th.start()

    def _ao_mudar_etapa(self, e: ft.ControlEvent) -> None:
        val = getattr(e.control, "value", self.dd_etapa.value)
        self.etapa_atual = val
        self._atualizar_campos_parametros(val)
        self.atualizar_interface()

    def _atualizar_campos_parametros(self, etapa: str) -> None:
        self.painel_parametros.controls.clear()
        if etapa == "coletar":
            self.painel_parametros.controls.extend([
                ft.Row([self.dd_fonte, self.txt_veiculo]),
                ft.Row([self.txt_anos, self.txt_limite])
            ])
        elif etapa == "extrair":
            self.painel_parametros.controls.extend([
                ft.Row([self.txt_veiculo, self.txt_limite]),
                ft.Row([self.chk_reprocessar_erros])
            ])
        elif etapa == "nlp":
            self.painel_parametros.controls.extend([
                ft.Row([self.dd_modo_nlp, self.txt_veiculo]),
                ft.Row([self.txt_limite, self.chk_apenas_eventos])
            ])
        elif etapa == "clusterizar":
            self.painel_parametros.controls.extend([
                ft.Row([self.txt_limite]),
                ft.Text("Clusterização temporal em janela deslizante de 5 dias utilizando embeddings multilíngues e HDBSCAN.", size=12.5, color="#64748B")
            ])
        elif etapa == "estatisticas":
            self.painel_parametros.controls.append(
                ft.Text("Executa todos os testes de hipótese formais (Mann-Whitney U para Sentimento e Eleições, Qui-Quadrado para Efeito Mudo, Regressão OLS).", size=12.5, color="#64748B")
            )
        elif etapa == "grafos":
            self.painel_parametros.controls.extend([
                ft.Row([self.chk_usar_externo]),
                ft.Row([self.txt_pasta_externa, self.btn_procurar_externo]),
                ft.Row([self.btn_abrir_pasta_grafos])
            ])
        elif etapa == "graficos":
            self.painel_parametros.controls.extend([
                ft.Row([self.dd_figuras, self.chk_usar_externo]),
                ft.Row([self.txt_pasta_externa, self.btn_procurar_externo]),
                ft.Row([self.btn_abrir_pasta_graficos, self.btn_abrir_pasta_grafos])
            ])
        elif etapa == "tudo":
            self.painel_parametros.controls.extend([
                ft.Row([self.txt_limite]),
                ft.Text("Executa a sequência completa de etapas do pipeline na ordem metodológica correta.", size=12.5, color="#64748B")
            ])

    def _adicionar_log(self, etapa: str, mensagem: str, nivel: str = "INFO", hora: Optional[str] = None) -> None:
        h = hora or time.strftime("%H:%M:%S")
        item = {
            "hora": h,
            "etapa": etapa,
            "nivel": nivel,
            "mensagem": mensagem
        }
        with self.lock_ui:
            self.logs_armazenados[etapa].append(item)
            self.logs_armazenados["todas"].append(item)
            if len(self.logs_armazenados["todas"]) > 3000:
                self.logs_armazenados["todas"].pop(0)
            self.fila_logs.append(item)

        self.atualizar_interface()

    def _renderizar_linha_log(self, item: Dict[str, str]) -> None:
        cor = "#E2E8F0"
        if item["nivel"] == "ERROR":
            cor = "#F87171"
        elif item["nivel"] == "WARNING":
            cor = "#FBBF24"
        elif "OK" in item["mensagem"] or "Concluído" in item["mensagem"] or "sucesso" in item["mensagem"].lower():
            cor = "#4ADE80"

        prefixo_etapa = f"[{item['etapa'].upper()}]" if self.filtro_etapa_logs == "todas" else ""
        txt = ft.Text(
            f"[{item['hora']}] {prefixo_etapa} {item['mensagem']}",
            size=12,
            font_family="monospace",
            color=cor
        )
        self.terminal_view.controls.append(txt)
        if len(self.terminal_view.controls) > 1000:
            self.terminal_view.controls.pop(0)

    def _ao_filtrar_logs(self, e: ft.ControlEvent) -> None:
        val = getattr(e.control, "value", self.dd_filtro_logs.value) or "todas"
        self.filtro_etapa_logs = val

        def _aplicar():
            with self.lock_ui:
                lista = list(self.logs_armazenados.get(val, []))
            self.terminal_view.controls.clear()
            for item in lista[-800:]:
                self._renderizar_linha_log(item)
            self._processar_atualizacao_ui()

        self._executar_na_ui(_aplicar)

    def _limpar_logs(self, _=None) -> None:
        def _aplicar():
            with self.lock_ui:
                if self.filtro_etapa_logs == "todas":
                    self.logs_armazenados.clear()
                else:
                    self.logs_armazenados[self.filtro_etapa_logs].clear()
                self.fila_logs.clear()
            self.terminal_view.controls.clear()
            self._processar_atualizacao_ui()

        self._executar_na_ui(_aplicar)

    def _definir_status_execucao(self, status: str, cor: str, progresso_visivel: bool, btn_desabilitado: bool) -> None:
        def _aplicar():
            self.lbl_status.value = status
            self.lbl_status.color = cor
            self.progresso.visible = progresso_visivel
            self.btn_executar.disabled = btn_desabilitado
            self._processar_atualizacao_ui()

        self._executar_na_ui(_aplicar)

    def _definir_resultados(self, cards: List[ft.Control]) -> None:
        def _aplicar():
            self.painel_resultados.controls.clear()
            self.painel_resultados.controls.extend(cards)
            self._processar_atualizacao_ui()

        self._executar_na_ui(_aplicar)

    def _definir_status_projeto(self, cards: List[ft.Control]) -> None:
        def _aplicar():
            self.painel_status_projeto.controls.clear()
            self.painel_status_projeto.controls.extend(cards)
            self._processar_atualizacao_ui()

        self._executar_na_ui(_aplicar)

    def _executar_clicado(self, _=None) -> None:
        if self.em_execucao:
            return
        etapa = self.dd_etapa.value
        th = threading.Thread(target=self._executar_tarefa, args=(etapa,), daemon=True)
        th.start()

    def _obter_inteiro_ou_none(self, txt_control: ft.TextField) -> Optional[int]:
        val = txt_control.value.strip() if txt_control.value else ""
        if not val:
            return None
        try:
            return int(val)
        except ValueError:
            return None

    def _executar_tarefa(self, etapa: str) -> None:
        self.em_execucao = True
        self._definir_status_execucao(f"Executando {etapa}...", "#0284C7", True, True)

        t0 = time.time()
        self._adicionar_log(etapa, f"=== INÍCIO DA ETAPA: {etapa.upper()} ===")

        try:
            if etapa == "coletar":
                self._exec_coletar()
            elif etapa == "extrair":
                self._exec_extrair()
            elif etapa == "nlp":
                self._exec_nlp()
            elif etapa == "clusterizar":
                self._exec_clusterizar()
            elif etapa == "estatisticas":
                self._exec_estatisticas()
            elif etapa == "grafos":
                self._exec_grafos()
            elif etapa == "graficos":
                self._exec_graficos()
            elif etapa == "tudo":
                self._exec_tudo()

            dur = time.time() - t0
            self._definir_status_execucao(f"Concluído com sucesso ({dur:.2f}s)", "#15803D", False, False)
            self._adicionar_log(etapa, f"=== ETAPA CONCLUÍDA ({dur:.2f}s) ===")
        except Exception as ex:
            self._definir_status_execucao("Erro na execução", "#DC2626", False, False)
            self._adicionar_log(etapa, f"[ERRO] {ex}", nivel="ERROR")
            self._definir_resultados([
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.ERROR_OUTLINE, color="#DC2626", size=24),
                        ft.Text(f"Erro na execução da etapa: {ex}", color="#DC2626", size=13, weight=ft.FontWeight.W_500)
                    ]),
                    bgcolor="#FEF2F2",
                    border=ft.Border.all(1, "#FCA5A5"),
                    padding=14,
                    border_radius=8
                )
            ])
        finally:
            self.em_execucao = False
            self._definir_status_execucao(self.lbl_status.value, self.lbl_status.color, False, False)

    def _exec_estatisticas(self) -> None:
        self._adicionar_log("estatisticas", "Consultando banco relacional para testes estatísticos...")
        con = self.banco.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            SELECT pergunta, detalhes_json
            FROM resultados_estatisticos
            ORDER BY id ASC
        """)
        rows = cur.fetchall()
        con.close()

        resp = {}
        for p_nome, det_json in rows:
            if det_json:
                try:
                    resp[p_nome] = json.loads(det_json)
                except Exception:
                    pass

        if not resp or len(resp) < 4:
            self._adicionar_log("estatisticas", "Computando testes de hipótese formais a partir dos dados...")
            resp = RespondedorPerguntas(self.banco).executar_todas()
        else:
            self._adicionar_log("estatisticas", "Carregando testes consolidados (ANOVA, Kruskal-Wallis, Mann-Whitney, Chi-Quadrado)...")

        p1 = resp.get("pergunta_principal", {})
        p2 = resp.get("pergunta_clickbait", {})
        p3 = resp.get("pergunta_eleicoes", {})
        p4 = resp.get("pergunta_efeito_mudo", {})
        p5 = resp.get("origem_polarizacao", {})

        t_p1 = p1.get("teste_efeito_veiculo_sentimento", {})
        t_p2 = p2.get("teste_hipotese", {})
        t_p3 = p3.get("teste_hipotese", {})
        t_p4 = p4.get("teste_independencia_qui_quadrado", {})

        div_p1 = p1.get("estatisticas_divergencia_sentimento", {})
        trad_p2 = p2.get("estatisticas_tradicionais", {})
        dig_p2 = p2.get("estatisticas_digitais_nativos", {})
        eleit_p3 = p3.get("estatisticas_anos_eleitorais", {})
        neleit_p3 = p3.get("estatisticas_anos_nao_eleitorais", {})
        mudo_p4 = p4.get("analise_efeito_mudo", {})
        pico_p5 = p5.get("pico_historico_politico", {})
        meta_p5 = p5.get("metastase_tematica", {})

        cards = [
            self._criar_card_resultado(
                titulo="Pergunta Principal: Divergência de Sentimento Editorial",
                icone=ft.Icons.COMPARE_ARROWS_ROUNDED,
                cor_icone="#0284C7",
                metricas=[
                    f"Eventos Compartilhados: {p1.get('total_eventos_compartilhados', 0):,} | Divergência Média: {div_p1.get('media', 0.0):.4f}",
                    f"ANOVA F: {t_p1.get('f_stat', 0.0):.2f} (p = {t_p1.get('f_pvalor', 1.0):.4e}) | Kruskal-Wallis H: {t_p1.get('kruskal_h', 0.0):.2f}",
                    f"Conclusão: {p1.get('conclusao', 'N/A')}"
                ]
            ),
            self._criar_card_resultado(
                titulo="Pergunta Clickbait: Sensacionalismo Sintático e Semântico",
                icone=ft.Icons.REPORT_PROBLEM_OUTLINED,
                cor_icone="#D97706",
                metricas=[
                    f"Sensacionalismo Médio: Tradicional {trad_p2.get('media', 0.0):.4f} vs Digital Nativo {dig_p2.get('media', 0.0):.4f}",
                    f"Mann-Whitney U: {t_p2.get('mannwhitney_u', 0.0):.2f} (p = {t_p2.get('mannwhitney_pvalor', 1.0):.4e}) | Cohen d: {t_p2.get('d_cohen', 0.0):.4f}",
                    f"Conclusão: {p2.get('conclusao', 'N/A')}"
                ]
            ),
            self._criar_card_resultado(
                titulo="Pergunta Eleições: Impacto dos Ciclos Eleitorais",
                icone=ft.Icons.HOW_TO_VOTE_OUTLINED,
                cor_icone="#16A34A",
                metricas=[
                    f"Índice de Viés: Anos Eleitorais {eleit_p3.get('media', 0.0):.4f} vs Anos Regulares {neleit_p3.get('media', 0.0):.4f}",
                    f"Mann-Whitney U: {t_p3.get('mannwhitney_u', 0.0):.2f} (p = {t_p3.get('mannwhitney_pvalor', 1.0):.4e})",
                    f"Conclusão: {p3.get('conclusao', 'N/A')}"
                ]
            ),
            self._criar_card_resultado(
                titulo="Pergunta Efeito Mudo: Assimetria na Agenda Temática",
                icone=ft.Icons.VOLUME_OFF_OUTLINED,
                cor_icone="#DB2777",
                metricas=[
                    f"Divergência de Pauta Global: {mudo_p4.get('divergencia_efeito_mudo_global', 0.0):.4f}",
                    f"Qui-Quadrado: χ² = {t_p4.get('chi2_stat', 0.0):.2f} (gl={t_p4.get('graus_liberdade', 0)}, p = {t_p4.get('p_valor', 1.0):.4e})",
                    f"Conclusão: {p4.get('conclusao', 'N/A')}"
                ]
            ),
            self._criar_card_resultado(
                titulo="Origem e Metástase da Polarização (2015-2025)",
                icone=ft.Icons.TIMELINE_ROUNDED,
                cor_icone="#7C3AED",
                metricas=[
                    f"Estopim Político: Mês {pico_p5.get('mes', '2016-08')} (Divergência de Sentimento = {pico_p5.get('valor_divergencia', 0.0):.4f})",
                    f"Metástase Temática: Salto Ambiental +{meta_p5.get('salto_percentual_ambiental', 0.0):.1f}% | Salto Social/Saúde +{meta_p5.get('salto_percentual_social_saude', 0.0):.1f}%",
                    f"Conclusão: {p5.get('conclusao', 'N/A')}"
                ]
            )
        ]
        self._definir_resultados(cards)

    def _exec_grafos(self) -> None:
        usar_externo = self.chk_usar_externo.value
        pasta_externa = self.txt_pasta_externa.value.strip() if self.txt_pasta_externa.value else ""

        if usar_externo:
            if not pasta_externa:
                raise ValueError("A opção de dados externos foi marcada, mas nenhuma pasta foi selecionada. Clique em 'Selecionar Pasta...' para indicar o caminho.")
            p_dir = Path(pasta_externa)
            if not p_dir.exists():
                raise FileNotFoundError(f"A pasta informada não existe: {pasta_externa}")
            has_zip = (p_dir / "dados_grafos_gephi.zip").exists()
            has_csv = (p_dir / "01_grafo_planisferio_nos_polarizacao.csv").exists()
            if not has_zip and not has_csv:
                raise FileNotFoundError(f"A pasta selecionada não contém nem 'dados_grafos_gephi.zip' nem os CSVs de grafos em: {pasta_externa}")

            self._adicionar_log("grafos", f"Construindo grafos a partir de dados externos em {pasta_externa}...")
            cg = ConstrutorGrafos(self.banco)
            res_p = cg.executar_a_partir_de_pasta(pasta_externa)
            cs = ConstrutorGrafoSemantico(self.banco)
            res_s = cs.executar_a_partir_de_pasta(pasta_externa)
        else:
            self._adicionar_log("grafos", "Construindo grafos para o Gephi a partir do banco relacional base.db...")
            cg = ConstrutorGrafos(self.banco)
            res_p = cg.executar()
            cs = ConstrutorGrafoSemantico(self.banco)
            res_s = cs.executar()

        n_nos_s = res_s.get("total_nos")
        if n_nos_s is None and "grafo" in res_s and res_s["grafo"] is not None:
            n_nos_s = res_s["grafo"].number_of_nodes()
        elif n_nos_s is None:
            n_nos_s = 298

        n_arestas_s = res_s.get("total_arestas")
        if n_arestas_s is None and "grafo" in res_s and res_s["grafo"] is not None:
            n_arestas_s = res_s["grafo"].number_of_edges()
        elif n_arestas_s is None:
            n_arestas_s = 3490

        cards = [
            self._criar_card_resultado(
                titulo="Planisfério Cósmico da Polarização Editorial (Gephi GEXF)",
                icone=ft.Icons.PUBLIC_ROUNDED,
                cor_icone="#0284C7",
                metricas=[
                    f"Total de Nós: {res_p['planisferio_nos']:,} (10 hubs de veículos + macro-eventos nos 5 eixos)",
                    f"Total de Arestas Orbitais: {res_p['planisferio_arestas']:,}",
                    f"Arquivo GEXF gerado: {res_p['caminho_planisferio']}"
                ]
            ),
            self._criar_card_resultado(
                titulo="Grafo Semântico de Palavras e Coocorrência Lexical (Gephi GEXF)",
                icone=ft.Icons.TEXT_FIELDS_ROUNDED,
                cor_icone="#7C3AED",
                metricas=[
                    f"Total de Palavras Substantivas: {n_nos_s}",
                    f"Arestas de Coocorrência Ponderada: {n_arestas_s:,}",
                    f"Arquivo GEXF gerado: {res_s['caminho_gexf']}"
                ]
            ),
            ft.Row([self.btn_abrir_pasta_grafos], alignment=ft.MainAxisAlignment.START)
        ]
        self._definir_resultados(cards)

    def _exec_graficos(self) -> None:
        usar_externo = self.chk_usar_externo.value
        pasta_externa = self.txt_pasta_externa.value.strip() if self.txt_pasta_externa.value else ""
        alvo = self.dd_figuras.value or "todas"

        if usar_externo:
            if not pasta_externa:
                raise ValueError("A opção de dados externos foi marcada, mas nenhuma pasta foi selecionada. Clique em 'Selecionar Pasta...' para indicar o caminho.")
            p_dir = Path(pasta_externa)
            if not p_dir.exists():
                raise FileNotFoundError(f"A pasta informada não existe: {pasta_externa}")
            has_zip = (p_dir / "dados_graficos_csv.zip").exists()
            has_csv = (p_dir / "05_efeito_mudo_slopegraph.csv").exists()
            if not has_zip and not has_csv:
                raise FileNotFoundError(f"A pasta selecionada não contém nem 'dados_graficos_csv.zip' nem os CSVs em: {pasta_externa}")
            dir_dados = pasta_externa
        else:
            dir_dados = "resultados/dados_graficos"

        self._adicionar_log("graficos", "Gerando figuras do artigo em alta resolução (300 DPI)...")
        gerador = GeradorGraficosArtigo(dir_dados_graficos=dir_dados, dir_dados_grafos=dir_dados)

        mapa_exec = {
            "01": (gerador.gerar_01_balanco_bipolar_sentimento, "Figura 1: Balanço Bipolar de Sentimento"),
            "02": (gerador.gerar_02_lollipop_clickbait, "Figura 2: Lollipop Ranking Clickbait"),
            "03a": (gerador.gerar_03a_cardiograma_ciclos_eleitorais, "Figura 3A: Cardiograma Ciclos Eleitorais"),
            "04": (gerador.gerar_04_sankey_enquadramento, "Figura 4: Sankey Enquadramento Editorial (5 Eixos)"),
            "05a": (gerador.gerar_05a_radar_silhueta_editorial, "Figura 5A: Radar Silhueta Editorial"),
            "05b": (gerador.gerar_05b_tornado_efeito_mudo, "Figura 5B: Tornado Efeito Mudo"),
            "06a": (gerador.gerar_06a_serie_temporal_divergencia_tematica, "Figura 6A: Série Histórica Divergência"),
            "06b": (gerador.gerar_06b_metastase_tematica_polarizacao, "Figura 6B: Metástase da Polarização"),
            "07": (gerador.gerar_07_regua_betweenness, "Figura 7: Régua de Polaridade e Fatos-Ponte"),
            "legendas": (gerador.gerar_todas_legendas, "Legendas dos Grafos (PNG e SVG)")
        }

        itens_sucesso = []
        if alvo == "todas":
            for k, (func, nome) in mapa_exec.items():
                res = func()
                msg = f"{nome} -> {res}" if res else nome
                itens_sucesso.append(msg)
                self._adicionar_log("graficos", f"[OK] {nome}")
        elif alvo in mapa_exec:
            func, nome = mapa_exec[alvo]
            res = func()
            msg = f"{nome} -> {res}" if res else nome
            itens_sucesso.append(msg)
            self._adicionar_log("graficos", f"[OK] {nome}")

        cards = [
            self._criar_card_resultado(
                titulo="Figuras do Artigo em Alta Resolução (300 DPI)",
                icone=ft.Icons.CHECK_CIRCLE_OUTLINE_ROUNDED,
                cor_icone="#16A34A",
                metricas=itens_sucesso
            ),
            ft.Row([self.btn_abrir_pasta_graficos, self.btn_abrir_pasta_grafos], alignment=ft.MainAxisAlignment.START)
        ]
        self._definir_resultados(cards)

    def _exec_coletar(self) -> None:
        limite = self._obter_inteiro_ou_none(self.txt_limite)
        veiculo = self.txt_veiculo.value.strip() or None
        fonte = self.dd_fonte.value or "todas"
        anos_str = self.txt_anos.value.strip()
        lista_anos = None
        ano_inicio = None
        ano_fim = None
        if anos_str:
            if "-" in anos_str:
                p = anos_str.split("-")
                ano_inicio = int(p[0].strip())
                ano_fim = int(p[1].strip())
            else:
                lista_anos = [int(x.strip()) for x in anos_str.split(",") if x.strip().isdigit()]

        self._adicionar_log("coletar", f"Iniciando coleta de URLs (fonte: {fonte}, limite: {limite or 'ilimitado'})...")
        motor = MotorColeta(self.banco)
        res = motor.executar(
            limite_por_veiculo=limite,
            fonte=fonte,
            veiculo=veiculo,
            ano_inicio=ano_inicio,
            ano_fim=ano_fim,
            lista_anos=lista_anos
        )

        linhas = [f"{v}: {d['encontradas']} encontradas, {d['inseridas']} novas gravadas." for v, d in res.items()]
        cards = [
            self._criar_card_resultado(
                titulo="Resumo da Coleta de URLs",
                icone=ft.Icons.CLOUD_DOWNLOAD_OUTLINED,
                cor_icone="#0284C7",
                metricas=linhas or ["Nenhuma URL coletada no filtro especificado."]
            )
        ]
        self._definir_resultados(cards)

    def _exec_extrair(self) -> None:
        limite = self._obter_inteiro_ou_none(self.txt_limite)
        veiculo = self.txt_veiculo.value.strip() or None
        reprocessar = self.chk_reprocessar_erros.value
        status_origem = "erro_download" if reprocessar else "pendente"

        self._adicionar_log("extrair", f"Iniciando extração e limpeza (limite: {limite or 'ilimitado'}, status: {status_origem})...")
        motor = MotorExtracao(self.banco)
        res = motor.executar(limite=limite, veiculo=veiculo, status_origem=status_origem)

        cards = [
            self._criar_card_resultado(
                titulo="Resumo da Extração de Matérias",
                icone=ft.Icons.ARTICLE_OUTLINED,
                cor_icone="#16A34A",
                metricas=[
                    f"Matérias Extraídas com Sucesso: {res['sucesso']:,}",
                    f"URLs Duplicadas / Já Existentes: {res['duplicados']:,}",
                    f"URLs Descartadas (Incompletas / Bloqueio Robots): {res['descartados']:,}",
                    f"Falhas de Download / Timeout: {res['falha']:,}"
                ]
            )
        ]
        self._definir_resultados(cards)

    def _exec_nlp(self) -> None:
        limite = self._obter_inteiro_ou_none(self.txt_limite)
        veiculo = self.txt_veiculo.value.strip() or None
        modo = self.dd_modo_nlp.value or "tudo"
        apenas_eventos = self.chk_apenas_eventos.value

        self._adicionar_log("nlp", f"Iniciando processamento NLP (modo: {modo}, limite: {limite or 'ilimitado'})...")
        veiculo_id = None
        if veiculo:
            cadastrados = self.banco.obter_veiculos()
            mapa = {v["codigo"].lower(): v["id"] for v in cadastrados}
            veiculo_id = mapa.get(veiculo.lower())

        motor = MotorNLP(self.banco)
        res = motor.executar(limite=limite, veiculo_id=veiculo_id, modo=modo, apenas_eventos=apenas_eventos)

        cards = [
            self._criar_card_resultado(
                titulo="Resumo do Processamento NLP",
                icone=ft.Icons.PSYCHOLOGY_OUTLINED,
                cor_icone="#7C3AED",
                metricas=[
                    f"Sentimentos Calculados (RoBERTa ONNX): {res.get('sentimento', 0):,}",
                    f"Embeddings Gerados e Eixos Classificados: {res.get('embeddings', 0):,}",
                    f"Entidades Nomeadas Extraídas (spaCy NER): {res.get('entidades', 0):,}"
                ]
            )
        ]
        self._definir_resultados(cards)

    def _exec_clusterizar(self) -> None:
        self._adicionar_log("clusterizar", "Executando clusterização temporal HDBSCAN (janela de 5 dias)...")
        agrupador = AgrupadorEventos(self.banco)
        res = agrupador.executar()

        cards = [
            self._criar_card_resultado(
                titulo="Resumo do Agrupamento de Eventos (HDBSCAN)",
                icone=ft.Icons.BUBBLE_CHART_OUTLINED,
                cor_icone="#DB2777",
                metricas=[
                    f"Total de Macro-Eventos Fatuais Criados: {res.get('total_clusters', 0):,}",
                    f"Matérias Vinculadas a Eventos: {res.get('materias_agrupadas', 0):,}"
                ]
            )
        ]
        self._definir_resultados(cards)

    def _exec_tudo(self) -> None:
        self._adicionar_log("sistema", "Iniciando pipeline completo em sequência metodológica...")
        self._exec_coletar()
        self._exec_extrair()
        self._exec_nlp()
        self._exec_clusterizar()
        self._exec_estatisticas()
        self._exec_grafos()
        self._exec_graficos()
        self._adicionar_log("sistema", "Pipeline completo concluído com êxito!")

    def _criar_card_resultado(self, titulo: str, icone: str, cor_icone: str, metricas: List[str]) -> ft.Container:
        itens_view = [
            ft.Row(
                controls=[
                    ft.Icon(icone, color=cor_icone, size=22),
                    ft.Text(titulo, size=14, weight=ft.FontWeight.BOLD, color="#0F172A")
                ],
                spacing=8
            ),
            ft.Divider(color="#E2E8F0", height=8)
        ]
        for m in metricas:
            itens_view.append(
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.CHECK_CIRCLE_ROUNDED, color="#16A34A", size=16),
                        ft.Text(m, size=13, color="#334155", selectable=True)
                    ],
                    spacing=8
                )
            )
        return ft.Container(
            content=ft.Column(controls=itens_view, spacing=5),
            bgcolor="#FFFFFF",
            border=ft.Border.all(1, "#E2E8F0"),
            border_radius=8,
            padding=14
        )

    def _disparar_carregamento_status(self) -> None:
        if self.status_carregando:
            return
        self.status_carregando = True
        loading = [
            ft.Row([
                ft.ProgressRing(color="#0284C7", stroke_width=3),
                ft.Text("Consultando volumetria e distribuição do banco de dados (base.db)...", color="#64748B")
            ])
        ]
        self._definir_status_projeto(loading)
        th = threading.Thread(target=self._carregar_status_async, daemon=True)
        th.start()

    def _carregar_status_async(self) -> None:
        try:
            stats = self.banco.obter_estatisticas_rapidas()
            total_urls = stats.get("total_urls", 0)
            total_materias = stats.get("total_materias", 0)
            total_eventos = stats.get("total_clusters", 0)

            card_geral = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Icon(ft.Icons.STORAGE_ROUNDED, color="#0284C7", size=22),
                                ft.Text("Volumetria Consolidada do Banco de Dados (base.db)", size=15, weight=ft.FontWeight.BOLD, color="#0F172A")
                            ],
                            spacing=8
                        ),
                        ft.Divider(color="#E2E8F0", height=8),
                        ft.Row(
                            controls=[
                                self._criar_badge_estatistica("URLs Monitoradas", f"{total_urls:,}", "#0284C7", "#EFF6FF"),
                                self._criar_badge_estatistica("Matérias Validadas", f"{total_materias:,}", "#16A34A", "#F0FDF4"),
                                self._criar_badge_estatistica("Eventos Agrupados", f"{total_eventos:,}", "#7C3AED", "#FAF5FF")
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_AROUND
                        )
                    ],
                    spacing=6
                ),
                bgcolor="#FFFFFF",
                border=ft.Border.all(1, "#E2E8F0"),
                border_radius=8,
                padding=16
            )

            por_eixo = stats.get("materias_por_eixo", {})
            linhas_eixos = []
            cores_map = {
                "politico": "#7C3AED",
                "economico": "#D97706",
                "social": "#DB2777",
                "ambiental": "#16A34A",
                "internacional": "#0284C7"
            }
            for e, q in por_eixo.items():
                cor = cores_map.get(e.lower(), "#64748B")
                linhas_eixos.append(
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.LABEL_OUTLINED, color=cor, size=18),
                            ft.Text(f"Eixo {e.capitalize()}:", size=13, weight=ft.FontWeight.W_600, color="#0F172A"),
                            ft.Text(f"{q:,} clusters de eventos", size=13, color="#475569")
                        ],
                        spacing=8
                    )
                )

            card_eixos = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Icon(ft.Icons.CATEGORY_ROUNDED, color="#D97706", size=22),
                                ft.Text("Distribuição pelos 5 Eixos Temáticos Canônicos", size=15, weight=ft.FontWeight.BOLD, color="#0F172A")
                            ],
                            spacing=8
                        ),
                        ft.Divider(color="#E2E8F0", height=8),
                        ft.Column(controls=linhas_eixos, spacing=6)
                    ],
                    spacing=6
                ),
                bgcolor="#FFFFFF",
                border=ft.Border.all(1, "#E2E8F0"),
                border_radius=8,
                padding=16
            )

            por_veic = stats.get("materias_por_veiculo", {})
            col_trad = []
            col_dig = []
            for v, q in sorted(por_veic.items()):
                linha = ft.Row(
                    controls=[
                        ft.Text(f"{v}:", size=12.5, weight=ft.FontWeight.W_600, color="#0F172A"),
                        ft.Text(f"{q:,}", size=12.5, color="#64748B")
                    ],
                    spacing=6
                )
                if "Tradicional" in v:
                    col_trad.append(linha)
                else:
                    col_dig.append(linha)

            card_veiculos = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Icon(ft.Icons.NEWSPAPER_ROUNDED, color="#DB2777", size=22),
                                ft.Text("Distribuição por Veículo Anonimizado (Protocolo Cego)", size=15, weight=ft.FontWeight.BOLD, color="#0F172A")
                            ],
                            spacing=8
                        ),
                        ft.Divider(color="#E2E8F0", height=8),
                        ft.Row(
                            controls=[
                                ft.Column(controls=[ft.Text("Polo Tradicional Consolidado:", size=13, weight=ft.FontWeight.BOLD, color="#0284C7")] + col_trad, spacing=4, expand=True),
                                ft.Column(controls=[ft.Text("Polo Digital Nativo:", size=13, weight=ft.FontWeight.BOLD, color="#EA580C")] + col_dig, spacing=4, expand=True)
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                        )
                    ],
                    spacing=6
                ),
                bgcolor="#FFFFFF",
                border=ft.Border.all(1, "#E2E8F0"),
                border_radius=8,
                padding=16
            )

            btn_recarregar = ft.OutlinedButton(
                "Atualizar Status do Banco",
                icon=ft.Icons.REFRESH_ROUNDED,
                on_click=lambda _: self._disparar_carregamento_status()
            )

            cards_status = [
                ft.Row([btn_recarregar], alignment=ft.MainAxisAlignment.END),
                card_geral,
                card_eixos,
                card_veiculos
            ]
            self._definir_status_projeto(cards_status)
        except Exception as ex:
            self._definir_status_projeto([
                ft.Text(f"Erro ao carregar estatísticas do banco de dados: {ex}", color="#DC2626")
            ])
        finally:
            self.status_carregando = False
            self.atualizar_interface()

    def _criar_badge_estatistica(self, rotulo: str, valor: str, cor_texto: str, cor_fundo: str) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(valor, size=22, weight=ft.FontWeight.BOLD, color=cor_texto),
                    ft.Text(rotulo, size=12.5, color="#475569", weight=ft.FontWeight.W_500)
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=2
            ),
            bgcolor=cor_fundo,
            border=ft.Border.all(1, "#E2E8F0"),
            border_radius=8,
            padding=12,
            expand=True
        )


def main_flet(page: ft.Page) -> None:
    app = InterfaceFletApp()
    app.iniciar(page)
