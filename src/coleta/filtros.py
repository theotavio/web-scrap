import re
from typing import Optional
from src.configuracao import obter_configuracao


class FiltroUrls:
    def __init__(self):
        self.config = obter_configuracao()
        self.padroes_excluir = self.config.obter("filtros.padroes_excluir", [])
        self.eixos = self.config.obter("filtros.eixos_tematicos", {})
        self.ano_inicio = int(self.config.obter("geral.data_inicio", "2015-01-01").split("-")[0])
        self.ano_fim = int(self.config.obter("geral.data_fim", "2025-12-31").split("-")[0])

    def eh_ano_valido(self, url: str, data_prevista: Optional[str] = None) -> bool:
        if data_prevista:
            match_d = re.search(r"(20\d{2})", str(data_prevista))
            if match_d:
                ano = int(match_d.group(1))
                if ano < self.ano_inicio or ano > self.ano_fim:
                    return False

        match_url = re.search(r"(?:/|-|_)(20\d{2})(?:/|-|_|\.|$)", url)
        if match_url:
            ano = int(match_url.group(1))
            if ano < self.ano_inicio or ano > self.ano_fim:
                return False

        return True

    def eh_url_valida(self, url: str, data_prevista: Optional[str] = None) -> bool:
        if not url or not url.startswith("http"):
            return False

        url_baixa = url.lower()
        for padrao in self.padroes_excluir:
            if padrao in url_baixa:
                return False

        if len(url) < 22:
            return False

        if not self.eh_ano_valido(url_baixa, data_prevista):
            return False

        indicadores_noticia = [
            "/noticia/", "/politica/", "/economia/", "/mundo/", "/brasil/",
            "/poder/", "/poder-", "/mercado/", "/cotidiano/", "/republica/", "/governo/",
            "/congresso/", "/justica/", "/eleicoes/", "/internacional/",
            "/colunas/", "/coluna/", "/opiniao/", "/artigo/", "/nacional/",
            "/vozes/", "/materias/", "/materia/", "/post/", "/analise/", "/ultimas-noticias/",
            "/natureza/", "/educacao/", "/saude/", "/expresso/", "/ensaio/",
            "/grafico/", "/tema/", "/ponto-de-vista/", "/entrevista/", "/academico/",
            "/externo/", "/extra/", "/midia/", "/europa/", "/blogs/", "/blog/",
            "/cidades-df/", "/distrito-federal/", "/reportagem/", "/especial/",
            "/app/noticia/", "/holofote/", "/direito-e-justica/",
            ".ghtml", ".shtml", ".html",
            "/2015/", "/2016/", "/2017/", "/2018/", "/2019/", "/2020/",
            "/2021/", "/2022/", "/2023/", "/2024/", "/2025/"
        ]
        tem_segmento_noticia = any(ind in url_baixa for ind in indicadores_noticia)
        return tem_segmento_noticia

    def classificar_eixo_tematico(self, url: str, texto: Optional[str] = None) -> str:
        conteudo = url.lower()
        if texto:
            conteudo += " " + texto.lower()

        pontos_eixo = {}
        for eixo, termos in self.eixos.items():
            pts = 0
            for termo in termos:
                if termo in conteudo:
                    pts += 1
            pontos_eixo[eixo] = pts

        melhor_eixo = max(pontos_eixo, key=pontos_eixo.get)
        if pontos_eixo[melhor_eixo] > 0:
            return melhor_eixo

        if any(w in conteudo for w in ["politica", "governo", "poder", "congresso", "senado", "stf", "camara"]):
            return "politico"
        if any(w in conteudo for w in ["economia", "mercado", "financas", "inflacao", "dolar", "selic", "pib"]):
            return "economico"
        if any(w in conteudo for w in ["mundo", "internacional", "guerra", "exterior", "global"]):
            return "internacional"
        if any(w in conteudo for w in ["esporte", "futebol", "copa", "atleta", "time"]):
            return "esportivo"
        if any(w in conteudo for w in ["ambiente", "amazonia", "pantanal", "clima", "natureza", "queimada"]):
            return "ambiental"
        if any(w in conteudo for w in ["saude", "educacao", "seguranca", "policia", "sus", "escola"]):
            return "social"

        return "politico"
