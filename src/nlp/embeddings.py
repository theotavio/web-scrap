from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from src.configuracao import obter_configuracao
from src.registro import obter_logger


class GeradorEmbeddings:
    _instancia = None
    _modelo = None

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(GeradorEmbeddings, cls).__new__(cls)
        return cls._instancia

    def __init__(self):
        if self._modelo is None:
            config = obter_configuracao()
            self.logger = obter_logger("nlp")
            nome_modelo = config.obter(
                "nlp.modelo_embeddings",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._modelo = SentenceTransformer(nome_modelo)

    def gerar_vetor(self, texto: str) -> np.ndarray:
        vetor = self._modelo.encode(texto, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vetor, dtype=np.float32)

    def gerar_vetores_em_lote(self, textos: List[str], batch_size: int = 32) -> np.ndarray:
        if not textos:
            return np.empty((0, 384), dtype=np.float32)
        vetores = self._modelo.encode(
            textos,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.array(vetores, dtype=np.float32)

    @staticmethod
    def serializar_vetor(vetor: np.ndarray) -> bytes:
        return vetor.astype(np.float32).tobytes()

    @staticmethod
    def desserializar_vetor(dados_bytes: bytes, dimensao: int = 384) -> np.ndarray:
        return np.frombuffer(dados_bytes, dtype=np.float32)
