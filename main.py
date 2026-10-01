import argparse
import sys
import warnings
from typing import Optional
warnings.filterwarnings("ignore")
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados
from src.interface import InterfaceConsole
from src.coleta.motor_coleta import MotorColeta
from src.extracao.motor_extracao import MotorExtracao
from src.nlp.motor_nlp import MotorNLP
from src.cluster.agrupador import AgrupadorEventos
from src.grafos.construtor_grafo import ConstrutorGrafos
from src.analise.perguntas import RespondedorPerguntas
from src.visualizacao.graficos import GeradorGraficos
from src.visualizacao.relatorio import GeradorRelatorio


def etapa_coletar(
    banco: BancoDados,
    limite_veiculo: Optional[int],
    fonte: str,
    veiculo: Optional[str],
    ano_inicio: Optional[int],
    ano_fim: Optional[int],
    ui: InterfaceConsole
) -> None:
    msg_limite = f"limite por veículo: {limite_veiculo}" if limite_veiculo else "modo contínuo sem limites"
    detalhes = []
    if veiculo:
        detalhes.append(f"veículo: {veiculo}")
    if ano_inicio or ano_fim:
        ini = ano_inicio or 2015
        fim = ano_fim or 2025
        detalhes.append(f"período: {ini}-{fim}")
    detalhes_str = f" [{', '.join(detalhes)}]" if detalhes else ""

    ui.exibir_mensagem(f"Iniciando coleta de URLs (fonte: {fonte}, {msg_limite}){detalhes_str}...", "cyan")
    motor = MotorColeta(banco)
    resumo = motor.executar(
        limite_por_veiculo=limite_veiculo,
        fonte=fonte,
        veiculo=veiculo,
        ano_inicio=ano_inicio,
        ano_fim=ano_fim
    )
    for veic, dados in resumo.items():
        ui.exibir_mensagem(f"{veic}: {dados['encontradas']} encontradas, {dados['inseridas']} novas registradas.")


def etapa_extrair(banco: BancoDados, limite: Optional[int], ui: InterfaceConsole) -> None:
    ui.exibir_mensagem("Iniciando extração e limpeza de matérias com cascata anti-paywall...", "cyan")
    motor = MotorExtracao(banco)
    pendentes = banco.obter_urls_pendentes(limite=limite)
    total = len(pendentes)
    if total == 0:
        ui.exibir_mensagem("Nenhuma URL pendente para extração.", "yellow")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=ui.console
    ) as progress:
        tarefa = progress.add_task("[green]Extraindo matérias...", total=total)
        res = motor.executar(limite=limite, callback_progresso=lambda n: progress.update(tarefa, advance=n))

    ui.exibir_mensagem(
        f"Extração concluída: {res['sucesso']} sucessos, {res['descartados']} descartados, "
        f"{res['duplicados']} duplicados, {res['falha']} falhas."
    )


def etapa_nlp(banco: BancoDados, ui: InterfaceConsole) -> None:
    ui.exibir_mensagem("Iniciando processamento NLP (Embeddings, Sentimento, Clickbait, NER)...", "cyan")
    motor = MotorNLP(banco)
    sem_emb = banco.obter_materias_sem_embedding()
    sem_feat = banco.obter_materias_para_nlp()

    total_ops = len(sem_emb) + len(sem_feat)
    if total_ops == 0:
        ui.exibir_mensagem("Todas as matérias já possuem embeddings e features calculadas.", "yellow")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=ui.console
    ) as progress:
        tarefa = progress.add_task("[magenta]Calculando vetores e features...", total=total_ops)
        res = motor.executar(callback_progresso=lambda n: progress.update(tarefa, advance=n))

    ui.exibir_mensagem(f"NLP concluído: {res['embeddings']} embeddings e {res['features']} features gerados.")


def etapa_clusterizar(banco: BancoDados, ui: InterfaceConsole) -> None:
    ui.exibir_mensagem("Iniciando clustering temporal de eventos com HDBSCAN...", "cyan")
    agrupador = AgrupadorEventos(banco)
    res = agrupador.executar()
    ui.exibir_mensagem(
        f"Clustering concluído: {res['total_clusters']} clusters factuais criados, "
        f"{res['materias_agrupadas']} matérias associadas a eventos."
    )


def etapa_grafos(banco: BancoDados, ui: InterfaceConsole) -> None:
    ui.exibir_mensagem("Construindo modelagem em grafos (Bipartido e Projeção)...", "cyan")
    construtor = ConstrutorGrafos(banco)
    res = construtor.executar()
    ui.exibir_mensagem(
        f"Grafos construídos: Bipartido ({res['bipartido_nos']} nós, {res['bipartido_arestas']} arestas) | "
        f"Projeção ({res['projecao_nos']} nós, {res['projecao_arestas']} arestas)."
    )
    ui.exibir_mensagem(f"Arquivos GEXF para Gephi salvos em: {res['caminho_gexf_projecao']}")


def etapa_estatisticas(banco: BancoDados, ui: InterfaceConsole) -> None:
    ui.exibir_mensagem("Executando análises estatísticas e testes de hipótese...", "cyan")
    respondedor = RespondedorPerguntas(banco)
    respostas = respondedor.executar_todas()
    ui.exibir_mensagem("Pergunta Principal: " + respostas["pergunta_principal"]["conclusao"], "bold green")
    ui.exibir_mensagem("Pergunta Clickbait: " + respostas["pergunta_clickbait"]["conclusao"], "bold green")
    ui.exibir_mensagem("Pergunta Eleições: " + respostas["pergunta_eleicoes"]["conclusao"], "bold green")
    ui.exibir_mensagem("Pergunta Efeito Mudo: " + respostas["pergunta_efeito_mudo"]["conclusao"], "bold green")


def etapa_relatorio(banco: BancoDados, ui: InterfaceConsole) -> None:
    ui.exibir_mensagem("Gerando visualizações interativas Plotly e relatórios finais...", "cyan")
    construtor = ConstrutorGrafos(banco)
    G_bip = construtor.construir_grafo_bipartido()
    G_proj = construtor.construir_projecao_veiculos(G_bip)

    gerador_graficos = GeradorGraficos(banco)
    caminhos_graficos = gerador_graficos.executar(G_proj)

    respondedor = RespondedorPerguntas(banco)
    respostas = respondedor.executar_todas()

    metricas_grafos = construtor.executar()
    stats_banco = banco.obter_estatisticas_gerais()

    gerador_relatorio = GeradorRelatorio(banco)
    caminho_md = gerador_relatorio.gerar_relatorio_markdown(stats_banco, respostas, metricas_grafos)

    ui.exibir_mensagem(f"Relatório final gerado: {caminho_md}")
    ui.exibir_mensagem("Gráficos interativos salvos em relatorios/graficos/")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Raspagem e Modelagem Relacional de Notícias Brasileiras"
    )
    parser.add_argument(
        "--etapa",
        choices=["coletar", "extrair", "nlp", "features", "clusterizar", "grafos", "estatisticas", "relatorio", "tudo"],
        help="Etapa específica a ser executada"
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Limite de URLs para coletar/extrair (padrão: ilimitado/contínuo)"
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Apaga e recria o banco de dados SQLite"
    )
    parser.add_argument(
        "--limpar-logs",
        action="store_true",
        help="Limpa e esvazia todos os arquivos de log"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Exibe estatísticas atuais do banco de dados"
    )
    parser.add_argument(
        "--fonte",
        choices=["todas", "sitemaps", "cdx"],
        default="todas",
        help="Fonte de coleta de URLs (sitemaps: direto dos jornais | cdx: Wayback Machine | todas: ambas)"
    )
    parser.add_argument(
        "--veiculo",
        type=str,
        default=None,
        help="Filtrar por veículo específico (ex: nexo, brasil247, metropoles, folha, estadao, etc.)"
    )
    parser.add_argument(
        "--ano",
        type=int,
        default=None,
        help="Filtrar coleta por um ano específico (ex: 2025, 2020)"
    )
    parser.add_argument(
        "--ano-inicio",
        type=int,
        default=None,
        help="Ano inicial para a coleta (padrão: 2015)"
    )
    parser.add_argument(
        "--ano-fim",
        type=int,
        default=None,
        help="Ano final para a coleta (padrão: 2025)"
    )

    args = parser.parse_args()
    ui = InterfaceConsole()
    ui.exibir_cabecalho()

    if args.limpar_logs:
        from src.registro import limpar_logs
        limpar_logs()
        ui.exibir_mensagem("Arquivos de log limpos com sucesso.", "bold green")
        if not args.etapa and not args.reset_db and not args.status:
            return

    banco = BancoDados()

    if args.reset_db:
        banco.resetar()
        ui.exibir_mensagem("Banco de dados SQLite resetado com sucesso.", "bold red")
        if not args.etapa:
            return

    if args.status:
        stats = banco.obter_estatisticas_gerais()
        ui.exibir_status(stats)
        return

    if not args.etapa:
        stats = banco.obter_estatisticas_gerais()
        ui.exibir_status(stats)
        ui.exibir_mensagem(
            "Nenhuma etapa especificada. Use --etapa [coletar|extrair|nlp|clusterizar|grafos|estatisticas|relatorio|tudo].",
            "yellow"
        )
        return

    ano_ini = args.ano if args.ano else args.ano_inicio
    ano_fim = args.ano if args.ano else args.ano_fim

    etapa = args.etapa
    if etapa in ("coletar", "tudo"):
        etapa_coletar(
            banco,
            limite_veiculo=args.limite,
            fonte=args.fonte,
            veiculo=args.veiculo,
            ano_inicio=ano_ini,
            ano_fim=ano_fim,
            ui=ui
        )

    if etapa in ("extrair", "tudo"):
        limite_ext = (args.limite * 10) if args.limite else None
        etapa_extrair(banco, limite=limite_ext, ui=ui)

    if etapa in ("nlp", "features", "tudo"):
        etapa_nlp(banco, ui=ui)

    if etapa in ("clusterizar", "tudo"):
        etapa_clusterizar(banco, ui=ui)

    if etapa in ("grafos", "tudo"):
        etapa_grafos(banco, ui=ui)

    if etapa in ("estatisticas", "tudo"):
        etapa_estatisticas(banco, ui=ui)

    if etapa in ("relatorio", "tudo"):
        etapa_relatorio(banco, ui=ui)

    stats_finais = banco.obter_estatisticas_gerais()
    ui.exibir_status(stats_finais)


if __name__ == "__main__":
    main()
