from collections import Counter
from typing import Dict, List, Tuple
import spacy
from src.configuracao import obter_configuracao
from src.registro import obter_logger


class ExtratorEntidades:
    _instancia = None
    _nlp = None

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(ExtratorEntidades, cls).__new__(cls)
        return cls._instancia

    def __init__(self):
        if self._nlp is None:
            config = obter_configuracao()
            self.logger = obter_logger("nlp")
            nome_modelo = config.obter("nlp.modelo_spacy", "pt_core_news_lg")
            try:
                self._nlp = spacy.load(nome_modelo, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
            except Exception:
                try:
                    self._nlp = spacy.load("pt_core_news_lg")
                except Exception as e:
                    self.logger.error(f"Erro ao carregar modelo spaCy {nome_modelo}: {e}")
                    self._nlp = spacy.blank("pt")

    def extrair_entidades(self, texto: str) -> List[Tuple[str, str, int]]:
        if not texto or not texto.strip():
            return []

        doc = self._nlp(texto[:4000])
        entidades = []
        for ent in doc.ents:
            rotulo = ent.label_
            if rotulo in ("PER", "PERSON", "ORG", "LOC"):
                texto_ent = ent.text.strip()
                if len(texto_ent) > 2 and not texto_ent.isnumeric():
                    tipo_norm = "PER" if rotulo in ("PER", "PERSON") else rotulo
                    entidades.append((texto_ent, tipo_norm))

        contagem = Counter(entidades)
        return [(ent[0], ent[1], cont) for ent, cont in contagem.most_common(15)]

    def extrair_em_lote(self, textos: List[str], batch_size: int = 32) -> List[List[Tuple[str, str, int]]]:
        if not textos:
            return []

        textos_truncados = [t[:4000] if t else "" for t in textos]
        docs = list(self._nlp.pipe(textos_truncados, batch_size=batch_size))
        resultados = []

        for doc in docs:
            entidades = []
            for ent in doc.ents:
                rotulo = ent.label_
                if rotulo in ("PER", "PERSON", "ORG", "LOC"):
                    texto_ent = ent.text.strip()
                    if len(texto_ent) > 2 and not texto_ent.isnumeric():
                        tipo_norm = "PER" if rotulo in ("PER", "PERSON") else rotulo
                        entidades.append((texto_ent, tipo_norm))

            contagem = Counter(entidades)
            resultados.append([(ent[0], ent[1], cont) for ent, cont in contagem.most_common(15)])

        return resultados
