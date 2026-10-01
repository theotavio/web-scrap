from typing import Dict, List
from pysentimiento import create_analyzer
from src.configuracao import obter_configuracao
from src.registro import obter_logger


class AnalisadorSentimento:
    _instancia = None
    _analisador = None

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(AnalisadorSentimento, cls).__new__(cls)
        return cls._instancia

    def __init__(self):
        if self._analisador is None:
            self.logger = obter_logger("nlp")
            try:
                self._analisador = create_analyzer(task="sentiment", lang="pt")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar analisador pysentimiento: {e}")
                self._analisador = None

    def analisar_texto(self, texto: str) -> Dict[str, float]:
        if not texto or not texto.strip():
            return {"polaridade": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        texto_truncado = texto[:512]
        if self._analisador:
            try:
                res = self._analisador.predict(texto_truncado)
                probas = res.probas
                pos = float(probas.get("POS", 0.0))
                neg = float(probas.get("NEG", 0.0))
                neu = float(probas.get("NEU", 0.0))
                polaridade = round(pos - neg, 4)
                return {
                    "polaridade": polaridade,
                    "pos": round(pos, 4),
                    "neg": round(neg, 4),
                    "neu": round(neu, 4)
                }
            except Exception as e:
                self.logger.error(f"Erro na predicao de sentimento: {e}")

        return self._analisar_heuristico(texto_truncado)

    def analisar_em_lote(self, textos: List[str]) -> List[Dict[str, float]]:
        if not textos:
            return []
        textos_truncados = [t[:512] if t else "" for t in textos]
        if self._analisador:
            try:
                predicoes = self._analisador.predict(textos_truncados)
                resultados = []
                for res in predicoes:
                    probas = res.probas
                    pos = float(probas.get("POS", 0.0))
                    neg = float(probas.get("NEG", 0.0))
                    neu = float(probas.get("NEU", 0.0))
                    polaridade = round(pos - neg, 4)
                    resultados.append({
                        "polaridade": polaridade,
                        "pos": round(pos, 4),
                        "neg": round(neg, 4),
                        "neu": round(neu, 4)
                    })
                return resultados
            except Exception as e:
                self.logger.error(f"Erro no processamento em lote de sentimento: {e}")

        return [self._analisar_heuristico(t) for t in textos_truncados]

    def _analisar_heuristico(self, texto: str) -> Dict[str, float]:
        texto_l = texto.lower()
        positivas = ["avanco", "crescimento", "sucesso", "vitoria", "melhora", "recorde", "positivo", "acordo", "aprovado"]
        negativas = ["crise", "queda", "morte", "corrupcao", "escandalo", "fracasso", "prejuizo", "violencia", "perigo"]

        pts_pos = sum(1 for p in positivas if p in texto_l)
        pts_neg = sum(1 for n in negativas if n in texto_l)
        total = pts_pos + pts_neg

        if total == 0:
            return {"polaridade": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        pos = round(pts_pos / (total + 1.0), 4)
        neg = round(pts_neg / (total + 1.0), 4)
        neu = round(1.0 - (pos + neg), 4)
        polaridade = round(pos - neg, 4)

        return {"polaridade": polaridade, "pos": pos, "neg": neg, "neu": neu}
