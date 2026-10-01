import re
from typing import List, Tuple
from src.configuracao import obter_configuracao
from src.nlp.sentimento import AnalisadorSentimento


class CalculadorAtribuicao:
    def __init__(self):
        config = obter_configuracao()
        self.tamanho_janela = config.obter("nlp.tamanho_janela_contexto_entidade", 30)
        self.analisador_sentimento = AnalisadorSentimento()

    def _extrair_janelas_contexto(self, texto: str, entidade: str) -> List[str]:
        palavras = re.findall(r"\w+|[^\w\s]", texto)
        entidade_tokens = re.findall(r"\w+|[^\w\s]", entidade)
        tamanho_ent = len(entidade_tokens)
        if tamanho_ent == 0 or len(palavras) == 0:
            return []

        janelas = []
        for i in range(len(palavras) - tamanho_ent + 1):
            fatia = palavras[i:i + tamanho_ent]
            if [t.lower() for t in fatia] == [t.lower() for t in entidade_tokens]:
                inicio = max(0, i - self.tamanho_janela)
                fim = min(len(palavras), i + tamanho_ent + self.tamanho_janela)
                janela_texto = " ".join(palavras[inicio:fim])
                janelas.append(janela_texto)

        return janelas

    def calcular_polaridade_entidade(self, texto: str, entidade: str) -> float:
        janelas = self._extrair_janelas_contexto(texto, entidade)
        if not janelas:
            return 0.0

        scores = []
        for j in janelas[:5]:
            res = self.analisador_sentimento.analisar_texto(j)
            scores.append(res["polaridade"])

        if not scores:
            return 0.0

        media_score = sum(scores) / len(scores)
        return round(float(media_score), 4)

    def processar_entidades_artigo(
        self,
        texto: str,
        entidades: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int, float]]:
        resultados = []
        for nome_ent, tipo_ent, contagem in entidades:
            pol = self.calcular_polaridade_entidade(texto, nome_ent)
            resultados.append((nome_ent, tipo_ent, contagem, pol))
        return resultados
