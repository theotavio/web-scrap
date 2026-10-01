from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from src.banco import BancoDados


class InterfaceConsole:
    def __init__(self):
        self.console = Console()

    def exibir_cabecalho(self, titulo: str = "QUANTIFICAÇÃO MULTIDIMENSIONAL DA COBERTURA JORNALÍSTICA") -> None:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(f"[bold cyan]{titulo}[/bold cyan]")
        grid.add_row("[dim]Pipeline de Coleta, NLP, Clustering, Grafos e Estatística (2015-2025)[/dim]")
        self.console.print(Panel(grid, border_style="cyan"))

    def exibir_status(self, stats: Dict[str, Any]) -> None:
        tabela_geral = Table(title="[bold green]Resumo Geral da Base de Dados SQLite[/bold green]", border_style="green")
        tabela_geral.add_column("Métrica do Pipeline", style="bold white")
        tabela_geral.add_column("Total Registrado", justify="right", style="cyan")

        status_urls = stats.get("status_urls", {})
        total_urls = stats.get("total_urls", 0)
        total_materias = stats.get("total_materias", 0)

        tabela_geral.add_row("Total de URLs Coletadas (CDX + Sitemaps)", str(total_urls))
        tabela_geral.add_row("  └ URLs Pendentes de Extração", str(status_urls.get("pendente", 0)))
        tabela_geral.add_row("  └ URLs Extraídas com Sucesso", str(status_urls.get("extraido", 0)))
        tabela_geral.add_row("  └ URLs Descartadas / Curtas / Falhas", str(status_urls.get("descartado", 0) + status_urls.get("falha", 0)))
        tabela_geral.add_row("  └ URLs com Conteúdo Duplicado (SHA-256)", str(status_urls.get("duplicado", 0)))
        tabela_geral.add_row("Total de Matérias Únicas no Banco", str(total_materias))
        tabela_geral.add_row("Embeddings Vetoriais Computados", str(stats.get("total_embeddings", 0)))
        tabela_geral.add_row("Clusters de Eventos Factuais (HDBSCAN)", str(stats.get("total_eventos", 0)))

        self.console.print(tabela_geral)

        veiculos_stats = stats.get("veiculos_stats", [])
        if veiculos_stats:
            tabela_veic = Table(
                title="[bold yellow]Distribuição de URLs e Matérias por Veículo e Grupo Editorial[/bold yellow]",
                border_style="yellow"
            )
            tabela_veic.add_column("Veículo", style="bold white")
            tabela_veic.add_column("Grupo Editorial", style="cyan")
            tabela_veic.add_column("URLs Coletadas", justify="right", style="yellow")
            tabela_veic.add_column("Matérias Extraídas", justify="right", style="magenta")
            tabela_veic.add_column("Aproveitamento", justify="right", style="green")

            for v in veiculos_stats:
                nome = v.get("nome", "")
                tipo_br = "Tradicional" if v.get("tipo") == "tradicional" else "Digital Nativo"
                u_cnt = v.get("total_urls", 0)
                m_cnt = v.get("total_materias", 0)
                taxa = f"{(m_cnt / u_cnt * 100):.1f}%" if u_cnt > 0 else "-"
                tabela_veic.add_row(nome, tipo_br, str(u_cnt), str(m_cnt), taxa)

            self.console.print(tabela_veic)

        urls_por_ano = stats.get("urls_por_ano", {})
        materias_por_ano = stats.get("materias_por_ano", {})
        todos_anos = sorted(list(set(list(urls_por_ano.keys()) + [str(a) for a in materias_por_ano.keys()])))

        if todos_anos:
            tabela_anos = Table(title="[bold blue]Distribuição Temporal por Ano / Período[/bold blue]", border_style="blue")
            tabela_anos.add_column("Ano / Período", style="bold white")
            tabela_anos.add_column("URLs Coletadas", justify="right", style="yellow")
            tabela_anos.add_column("Matérias Extraídas", justify="right", style="cyan")

            for a in todos_anos:
                u_qtd = urls_por_ano.get(a, 0)
                m_qtd = materias_por_ano.get(int(a), 0) if a.isdigit() else materias_por_ano.get(a, 0)
                tabela_anos.add_row(str(a), str(u_qtd), str(m_qtd))

            self.console.print(tabela_anos)

    def exibir_mensagem(self, texto: str, estilo: str = "green") -> None:
        self.console.print(f"[{estilo}]● {texto}[/{estilo}]")

    def exibir_erro(self, texto: str) -> None:
        self.console.print(f"[bold red]✖ ERRO:[/bold red] {texto}")
