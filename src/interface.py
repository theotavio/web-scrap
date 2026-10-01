from typing import Any, Dict, List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class InterfaceConsole:
    def __init__(self):
        self.console = Console()

    def _fmt_num(self, valor: int) -> str:
        if valor == 0:
            return "0"
        return f"{valor:,}".replace(",", ".")

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
        total_cdx = stats.get("total_urls_cdx", 0)
        total_sitemaps = stats.get("total_urls_sitemaps", 0)
        total_materias = stats.get("total_materias", 0)

        tabela_geral.add_row("Total de URLs Coletadas", self._fmt_num(total_urls))
        tabela_geral.add_row("  ├─ URLs via CDX (Wayback Machine)", f"[yellow]{self._fmt_num(total_cdx)}[/yellow]")
        tabela_geral.add_row("  └─ URLs via Sitemaps / RSS dos Portais", f"[blue]{self._fmt_num(total_sitemaps)}[/blue]")
        tabela_geral.add_row("  ├─ URLs Pendentes de Extração", self._fmt_num(status_urls.get("pendente", 0)))
        tabela_geral.add_row("  ├─ URLs Extraídas com Sucesso", self._fmt_num(status_urls.get("extraido", 0)))
        tabela_geral.add_row("  ├─ URLs Descartadas / Curtas / Falhas", self._fmt_num(status_urls.get("descartado", 0) + status_urls.get("falha", 0)))
        tabela_geral.add_row("  └─ URLs com Conteúdo Duplicado (SHA-256)", self._fmt_num(status_urls.get("duplicado", 0)))
        tabela_geral.add_row("Total de Matérias Únicas no Banco", f"[bold green]{self._fmt_num(total_materias)}[/bold green]")
        tabela_geral.add_row("Embeddings Vetoriais Computados", self._fmt_num(stats.get("total_embeddings", 0)))
        tabela_geral.add_row("Clusters de Eventos Factuais (HDBSCAN)", self._fmt_num(stats.get("total_eventos", 0)))

        self.console.print(tabela_geral)

        veiculos_stats = stats.get("veiculos_stats", [])
        if veiculos_stats:
            tabela_veic = Table(
                title="[bold yellow]Distribuição de URLs e Matérias por Veículo e Fonte[/bold yellow]",
                border_style="yellow"
            )
            tabela_veic.add_column("Veículo", style="bold white")
            tabela_veic.add_column("Grupo Editorial", style="cyan")
            tabela_veic.add_column("URLs CDX", justify="right", style="yellow")
            tabela_veic.add_column("URLs Sitemaps", justify="right", style="blue")
            tabela_veic.add_column("Total URLs", justify="right", style="bold white")
            tabela_veic.add_column("Matérias Extraídas", justify="right", style="magenta")
            tabela_veic.add_column("Aproveitamento", justify="right", style="green")

            soma_cdx = 0
            soma_sm = 0
            soma_tot = 0
            soma_mat = 0

            for v in veiculos_stats:
                nome = v.get("nome", "")
                tipo_br = "Tradicional" if v.get("tipo") == "tradicional" else "Digital Nativo"
                u_cdx = v.get("urls_cdx", 0)
                u_sm = v.get("urls_sitemaps", 0)
                u_cnt = v.get("total_urls", 0)
                m_cnt = v.get("total_materias", 0)
                taxa = f"{(m_cnt / u_cnt * 100):.1f}%" if u_cnt > 0 else "-"

                soma_cdx += u_cdx
                soma_sm += u_sm
                soma_tot += u_cnt
                soma_mat += m_cnt

                tabela_veic.add_row(
                    nome,
                    tipo_br,
                    self._fmt_num(u_cdx),
                    self._fmt_num(u_sm),
                    self._fmt_num(u_cnt),
                    self._fmt_num(m_cnt),
                    taxa
                )

            taxa_global = f"{(soma_mat / soma_tot * 100):.1f}%" if soma_tot > 0 else "-"
            tabela_veic.add_section()
            tabela_veic.add_row(
                "[bold]TOTAL GERAL[/bold]",
                "-",
                f"[bold yellow]{self._fmt_num(soma_cdx)}[/bold yellow]",
                f"[bold blue]{self._fmt_num(soma_sm)}[/bold blue]",
                f"[bold white]{self._fmt_num(soma_tot)}[/bold white]",
                f"[bold magenta]{self._fmt_num(soma_mat)}[/bold magenta]",
                f"[bold green]{taxa_global}[/bold green]"
            )

            self.console.print(tabela_veic)

            tabela_anos_p1 = Table(
                title="[bold magenta]Detalhamento Anual de URLs por Veículo — Período 2015 a 2020[/bold magenta]",
                border_style="magenta"
            )
            tabela_anos_p1.add_column("Veículo", style="bold white", no_wrap=True)
            for y in range(2015, 2021):
                tabela_anos_p1.add_column(str(y), justify="right", style="cyan")
            tabela_anos_p1.add_column("Subtotal (15-20)", justify="right", style="bold yellow")

            for v in veiculos_stats:
                nome = v.get("nome", "")
                anos_v = v.get("anos", {})
                valores_p1 = [anos_v.get(str(y), 0) for y in range(2015, 2021)]
                subtot_p1 = sum(valores_p1)
                tabela_anos_p1.add_row(
                    nome,
                    *[self._fmt_num(val) if val > 0 else "-" for val in valores_p1],
                    self._fmt_num(subtot_p1)
                )

            self.console.print(tabela_anos_p1)

            tabela_anos_p2 = Table(
                title="[bold magenta]Detalhamento Anual de URLs por Veículo — Período 2021 a 2025[/bold magenta]",
                border_style="magenta"
            )
            tabela_anos_p2.add_column("Veículo", style="bold white", no_wrap=True)
            for y in range(2021, 2026):
                tabela_anos_p2.add_column(str(y), justify="right", style="cyan")
            tabela_anos_p2.add_column("Indef.", justify="right", style="dim")
            tabela_anos_p2.add_column("Total Geral", justify="right", style="bold green")

            for v in veiculos_stats:
                nome = v.get("nome", "")
                anos_v = v.get("anos", {})
                valores_p2 = [anos_v.get(str(y), 0) for y in range(2021, 2026)]
                indef = anos_v.get("Indefinido", 0)
                tot_v = v.get("total_urls", 0)
                tabela_anos_p2.add_row(
                    nome,
                    *[self._fmt_num(val) if val > 0 else "-" for val in valores_p2],
                    self._fmt_num(indef) if indef > 0 else "-",
                    self._fmt_num(tot_v)
                )

            self.console.print(tabela_anos_p2)

        anos_stats = stats.get("anos_stats", {})
        if anos_stats:
            tabela_anos_geral = Table(
                title="[bold blue]Distribuição Temporal Geral por Ano e Fonte de Coleta[/bold blue]",
                border_style="blue"
            )
            tabela_anos_geral.add_column("Ano / Período", style="bold white")
            tabela_anos_geral.add_column("URLs CDX", justify="right", style="yellow")
            tabela_anos_geral.add_column("URLs Sitemaps", justify="right", style="blue")
            tabela_anos_geral.add_column("Total URLs", justify="right", style="bold white")
            tabela_anos_geral.add_column("Matérias Extraídas", justify="right", style="magenta")

            chaves_ordenadas = sorted(
                list(anos_stats.keys()),
                key=lambda x: (0, int(x)) if x.isdigit() else (1, x)
            )

            for a in chaves_ordenadas:
                info_a = anos_stats[a]
                u_cdx = info_a.get("cdx", 0)
                u_sm = info_a.get("sitemaps", 0)
                u_tot = info_a.get("total_urls", 0)
                m_qtd = info_a.get("materias", 0)

                tabela_anos_geral.add_row(
                    str(a),
                    self._fmt_num(u_cdx),
                    self._fmt_num(u_sm),
                    self._fmt_num(u_tot),
                    self._fmt_num(m_qtd)
                )

            self.console.print(tabela_anos_geral)

    def exibir_mensagem(self, texto: str, estilo: str = "green") -> None:
        self.console.print(f"[{estilo}]● {texto}[/{estilo}]")

    def exibir_erro(self, texto: str) -> None:
        self.console.print(f"[bold red]✖ ERRO:[/bold red] {texto}")
