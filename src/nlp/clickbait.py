import re
import unicodedata
from typing import Dict
from src.configuracao import obter_configuracao


class DetectorClickbait:
    def __init__(self):
        config = obter_configuracao()
        self.gatilhos = config.obter("clickbait.gatilhos_sensacionalistas", [])

    def _normalizar_ascii(self, texto: str) -> str:
        return unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8").lower()

    def calcular_features(self, titulo: str) -> Dict[str, float]:
        if not titulo:
            return {
                "proporcao_maiusculas": 0.0,
                "pontuacao_enfase": 0.0,
                "tem_numeros": 0.0,
                "qtd_gatilhos": 0.0,
                "score_clickbait": 0.0
            }

        letras = [c for c in titulo if c.isalpha()]
        maiusculas = [c for c in letras if c.isupper()]
        prop_maiusculas = len(maiusculas) / max(1, len(letras))

        pontos_enfase = len(re.findall(r"[!?]{1,}|\.{3,}", titulo))
        prop_pontuacao = min(1.0, pontos_enfase / 3.0)

        tem_num = 1.0 if re.search(r"\b\d+\b", titulo) else 0.0

        titulo_norm = self._normalizar_ascii(titulo)
        qtd_gat = 0
        for g in self.gatilhos:
            g_norm = self._normalizar_ascii(g)
            if re.search(r"\b" + re.escape(g_norm) + r"\b", titulo_norm):
                qtd_gat += 1

        prop_gatilhos = min(1.0, qtd_gat / 2.0)

        score = (
            0.30 * prop_maiusculas
            + 0.25 * prop_pontuacao
            + 0.15 * tem_num
            + 0.30 * prop_gatilhos
        )
        score_final = round(min(1.0, max(0.0, score)), 4)

        return {
            "proporcao_maiusculas": round(prop_maiusculas, 4),
            "pontuacao_enfase": round(prop_pontuacao, 4),
            "tem_numeros": tem_num,
            "qtd_gatilhos": float(qtd_gat),
            "score_clickbait": score_final
        }

    def calcular_score(self, titulo: str) -> float:
        return self.calcular_features(titulo)["score_clickbait"]
