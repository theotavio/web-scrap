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
    lista_anos: Optional[list],
    ui: InterfaceConsole
) -> None:
    msg_limite = f"limite por veículo: {limite_veiculo}" if limite_veiculo else "modo contínuo sem limites"
    detalhes = []
    if veiculo:
        detalhes.append(f"veículo(s): {veiculo}")
    if lista_anos:
        detalhes.append(f"ano(s): {', '.join(map(str, lista_anos))}")
    elif ano_inicio or ano_fim:
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
        ano_fim=ano_fim,
        lista_anos=lista_anos
    )
    for veic, dados in resumo.items():
        ui.exibir_mensagem(f"{veic}: {dados['encontradas']} encontradas, {dados['inseridas']} novas registradas.")


def etapa_extrair(
    banco: BancoDados,
    limite: Optional[int],
    veiculo: Optional[str],
    reprocessar_erros: bool = False,
    ui: InterfaceConsole = None
) -> None:
    status_alvo = "erro_download" if reprocessar_erros else "pendente"
    msg_detalhes = []
    if veiculo:
        msg_detalhes.append(f"veículo: {veiculo}")
    if limite:
        msg_detalhes.append(f"limite: {limite}")
    if reprocessar_erros:
        msg_detalhes.append("reprocessando erro_download")
    detalhes_str = f" [{', '.join(msg_detalhes)}]" if msg_detalhes else ""

    ui.exibir_mensagem(f"Iniciando extração e limpeza de matérias com cascata anti-paywall{detalhes_str}...", "cyan")
    motor = MotorExtracao(banco)

    veiculo_id = None
    if veiculo:
        mapa = motor._obter_mapa_veiculos()
        veiculo_id = mapa.get(veiculo.lower())

    total_pendentes = banco.contar_urls_pendentes(veiculo_id=veiculo_id, status=status_alvo)
    if total_pendentes == 0:
        ui.exibir_mensagem(f"Nenhuma URL com status '{status_alvo}' para extração.", "yellow")
        return

    total = min(limite, total_pendentes) if limite else total_pendentes

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
        res = motor.executar(
            limite=limite,
            veiculo=veiculo,
            status_origem=status_alvo,
            callback_progresso=lambda n: progress.update(tarefa, advance=n)
        )

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
        help="Filtrar por veículo(s) (ex: nexo | poder360,folha,estadao | tradicional | digital)"
    )
    parser.add_argument(
        "--ano",
        type=str,
        default=None,
        help="Filtrar por ano(s) (ex: 2020 | 2018,2020,2022 | 2018-2022)"
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
    parser.add_argument(
        "--reprocessar-erros",
        action="store_true",
        help="Reprocessa na etapa de extração apenas as URLs que falharam com erro_download"
    )
    parser.add_argument(
        "--reset-erros",
        action="store_true",
        help="Redefine todas as URLs com erro_download de volta para pendente no banco"
    )

    args = parser.parse_args()
    ui = InterfaceConsole()
    ui.exibir_cabecalho()

    if args.limpar_logs:
        from src.registro import limpar_logs
        limpar_logs()
        ui.exibir_mensagem("Arquivos de log limpos com sucesso.", "bold green")
        if not args.etapa and not args.reset_db and not args.status and not args.reset_erros:
            return

    banco = BancoDados()

    if args.reset_db:
        banco.resetar()
        ui.exibir_mensagem("Banco de dados SQLite resetado com sucesso.", "bold red")
        if not args.etapa:
            return

    if args.reset_erros:
        veiculo_id = None
        if args.veiculo:
            motor_temp = MotorExtracao(banco)
            mapa = motor_temp._obter_mapa_veiculos()
            veiculo_id = mapa.get(args.veiculo.lower())
        total_reset = banco.resetar_status_urls(status_origem="erro_download", status_destino="pendente", veiculo_id=veiculo_id)
        ui.exibir_mensagem(f"{total_reset} URLs com 'erro_download' foram redefinidas para 'pendente' com sucesso.", "bold green")
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

    ano_ini = args.ano_inicio
    ano_fim = args.ano_fim
    lista_anos = None

    if args.ano:
        ano_str = str(args.ano).strip()
        if "-" in ano_str and "," not in ano_str:
            partes = ano_str.split("-")
            if len(partes) == 2 and partes[0].isdigit() and partes[1].isdigit():
                ano_ini = int(partes[0])
                ano_fim = int(partes[1])
        elif "," in ano_str:
            lista_anos = [int(a.strip()) for a in ano_str.split(",") if a.strip().isdigit()]
        elif ano_str.isdigit():
            lista_anos = [int(ano_str)]

    etapa = args.etapa
    if etapa in ("coletar", "tudo"):
        etapa_coletar(
            banco,
            limite_veiculo=args.limite,
            fonte=args.fonte,
            veiculo=args.veiculo,
            ano_inicio=ano_ini,
            ano_fim=ano_fim,
            lista_anos=lista_anos,
            ui=ui
        )

    if etapa in ("extrair", "tudo"):
        etapa_extrair(
            banco,
            limite=args.limite,
            veiculo=args.veiculo,
            reprocessar_erros=args.reprocessar_erros,
            ui=ui
        )

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
