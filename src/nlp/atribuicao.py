import re
from typing import Dict, List, Optional, Tuple
from src.configuracao import obter_configuracao
from src.nlp.sentimento import AnalisadorSentimento


class CalculadorAtribuicao:
    def __init__(self):
        config = obter_configuracao()
        self.tamanho_janela = config.obter("nlp.tamanho_janela_contexto_entidade", 30)
        self.max_janelas = config.obter("nlp.max_janelas_por_entidade", 1)
        self.max_entidades = config.obter("nlp.max_entidades_atribuicao", 3)
        self.analisador_sentimento = AnalisadorSentimento()

    def _extrair_janelas_contexto(self, texto: str, entidade: str, max_janelas: Optional[int] = None) -> List[str]:
        palavras = re.findall(r"\w+|[^\w\s]", texto)
        palavras_lower = [p.lower() for p in palavras]
        entidade_tokens_lower = [t.lower() for t in re.findall(r"\w+|[^\w\s]", entidade)]
        tamanho_ent = len(entidade_tokens_lower)
        num_palavras = len(palavras)
        limite_janelas = max_janelas if max_janelas is not None else self.max_janelas

        if tamanho_ent == 0 or num_palavras == 0:
            return []

        janelas = []
        for i in range(num_palavras - tamanho_ent + 1):
            if palavras_lower[i:i + tamanho_ent] == entidade_tokens_lower:
                inicio = max(0, i - self.tamanho_janela)
                fim = min(num_palavras, i + tamanho_ent + self.tamanho_janela)
                janela_texto = " ".join(palavras[inicio:fim])
                janelas.append(janela_texto)
                if len(janelas) == limite_janelas:
                    break

        return janelas

    def calcular_polaridade_entidade(self, texto: str, entidade: str) -> float:
        janelas = self._extrair_janelas_contexto(texto, entidade)
        if not janelas:
            return 0.0

        resultados = self.analisador_sentimento.analisar_em_lote(janelas)
        scores = [r["polaridade"] for r in resultados]
        if not scores:
            return 0.0

        media_score = sum(scores) / len(scores)
        return round(float(media_score), 4)

    def processar_entidades_artigo(
        self,
        texto: str,
        entidades: List[Tuple[str, str, int]],
        polaridade_padrao: float = 0.0
    ) -> List[Tuple[str, str, int, float]]:
        if not entidades:
            return []

        palavras = re.findall(r"\w+|[^\w\s]", texto)
        palavras_lower = [p.lower() for p in palavras]
        num_palavras = len(palavras)

        ent_dados = []
        set_janelas = set()

        for idx_ent, (nome_ent, tipo_ent, contagem) in enumerate(entidades):
            janelas = []
            if tipo_ent in ("PER", "ORG") and idx_ent < self.max_entidades:
                entidade_tokens_lower = [t.lower() for t in re.findall(r"\w+|[^\w\s]", nome_ent)]
                tamanho_ent = len(entidade_tokens_lower)
                if tamanho_ent > 0 and num_palavras > 0:
                    for i in range(num_palavras - tamanho_ent + 1):
                        if palavras_lower[i:i + tamanho_ent] == entidade_tokens_lower:
                            inicio = max(0, i - self.tamanho_janela)
                            fim = min(num_palavras, i + tamanho_ent + self.tamanho_janela)
                            janela_texto = " ".join(palavras[inicio:fim])
                            janelas.append(janela_texto)
                            if len(janelas) == self.max_janelas:
                                break
                for j in janelas:
                    set_janelas.add(j)
            ent_dados.append((nome_ent, tipo_ent, contagem, janelas))

        mapa_polaridade = {}
        lista_janelas = list(set_janelas)
        if lista_janelas:
            resultados = self.analisador_sentimento.analisar_em_lote(lista_janelas)
            for j, r in zip(lista_janelas, resultados):
                mapa_polaridade[j] = r["polaridade"]

        resultados_finais = []
        for nome_ent, tipo_ent, contagem, janelas in ent_dados:
            if janelas:
                scores = [mapa_polaridade[j] for j in janelas if j in mapa_polaridade]
                media = round(float(sum(scores) / len(scores)), 4) if scores else polaridade_padrao
            else:
                media = polaridade_padrao
            resultados_finais.append((nome_ent, tipo_ent, contagem, media))

        return resultados_finais

    def processar_lote_artigos(
        self,
        lote_artigos: List[Tuple[int, str, List[Tuple[str, str, int]], float]]
    ) -> List[Tuple[int, List[Tuple[str, str, int, float]]]]:
        dados_artigos = []
        set_janelas = set()

        for m_id, corpo, ents, pol_materia in lote_artigos:
            if not ents:
                continue

            palavras = re.findall(r"\w+|[^\w\s]", corpo)
            palavras_lower = [p.lower() for p in palavras]
            num_palavras = len(palavras)

            ent_dados = []
            for idx_ent, (nome_ent, tipo_ent, contagem) in enumerate(ents):
                janelas = []
                if tipo_ent in ("PER", "ORG") and idx_ent < self.max_entidades:
                    entidade_tokens_lower = [t.lower() for t in re.findall(r"\w+|[^\w\s]", nome_ent)]
                    tamanho_ent = len(entidade_tokens_lower)
                    if tamanho_ent > 0 and num_palavras > 0:
                        for i in range(num_palavras - tamanho_ent + 1):
                            if palavras_lower[i:i + tamanho_ent] == entidade_tokens_lower:
                                inicio = max(0, i - self.tamanho_janela)
                                fim = min(num_palavras, i + tamanho_ent + self.tamanho_janela)
                                janela_texto = " ".join(palavras[inicio:fim])
                                janelas.append(janela_texto)
                                if len(janelas) == self.max_janelas:
                                    break
                    for j in janelas:
                        set_janelas.add(j)
                ent_dados.append((nome_ent, tipo_ent, contagem, janelas, pol_materia))
            dados_artigos.append((m_id, ent_dados))

        mapa_polaridade = {}
        lista_janelas = list(set_janelas)
        if lista_janelas:
            resultados = self.analisador_sentimento.analisar_em_lote(lista_janelas)
            for j, r in zip(lista_janelas, resultados):
                mapa_polaridade[j] = r["polaridade"]

        resultados_finais = []
        for m_id, ent_dados in dados_artigos:
            ent_atrib = []
            for nome_ent, tipo_ent, contagem, janelas, pol_materia in ent_dados:
                if janelas:
                    scores = [mapa_polaridade[j] for j in janelas if j in mapa_polaridade]
                    media = round(float(sum(scores) / len(scores)), 4) if scores else pol_materia
                else:
                    media = pol_materia
                ent_atrib.append((nome_ent, tipo_ent, contagem, media))
            resultados_finais.append((m_id, ent_atrib))

        return resultados_finais
