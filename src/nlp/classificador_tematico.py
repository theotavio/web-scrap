from typing import Dict, List, Optional
import numpy as np
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.nlp.embeddings import GeradorEmbeddings


class ClassificadorTematicoSemantico:
    _instancia = None
    _ancoras_carregadas = False

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(ClassificadorTematicoSemantico, cls).__new__(cls)
        return cls._instancia

    def __init__(self):
        if not self._ancoras_carregadas:
            self.config = obter_configuracao()
            self.logger = obter_logger("nlp")
            self.gerador = GeradorEmbeddings()
            self.rotulos: List[str] = []
            self.matriz_ancoras: Optional[np.ndarray] = None
            self._carregar_ancoras()
            self.__class__._ancoras_carregadas = True

    def _carregar_ancoras(self) -> None:
        ancoras_cfg = self.config.obter("classificacao_tematica.ancoras", {})
        if not ancoras_cfg:
            ancoras_cfg = {
                "politico": [
                    "eleições, votação, congresso nacional, senado federal, câmara dos deputados e partidos",
                    "governo federal, presidente da república, ministros, palácio do planalto e articulação política",
                    "decisões judiciais do supremo tribunal federal, stf, cpi e investigações parlamentares"
                ],
                "economico": [
                    "taxa de juros selic, comitê de política monetária copom e banco central",
                    "inflação, índice de preços ao consumidor ipca, custo de vida e poder de compra",
                    "mercado financeiro, bolsa de valores ibovespa, cotação do dólar, pib e arrecadação da fazenda"
                ],
                "social": [
                    "saúde pública, hospitais, sistema único de saúde sus, vacinação e atendimento médico",
                    "educação básica, escolas públicas, universidades federais, enem e professores",
                    "segurança pública, violência urbana, operações policiais, presídios e direitos humanos",
                    "programas sociais de combate à pobreza, fome e transferência de renda"
                ],
                "ambiental": [
                    "desmatamento ilegal na floresta amazônica, queimadas no pantanal e fiscalização do ibama",
                    "mudanças climáticas, aquecimento global, sustentabilidade, biodiversidade e transição energética"
                ],
                "internacional": [
                    "geopolítica global, diplomacia, relações exteriores, onu, g20 e brics",
                    "guerras, conflitos internacionais, tratados multilaterais e política externa dos estados unidos, rússia e china"
                ],
                "esportivo": [
                    "campeonato brasileiro de futebol, libertadores, copa do mundo e clubes esportivos",
                    "jogos olímpicos, atletas, competições esportivas e seleções nacionais"
                ]
            }

        rotulos = []
        vetores_centroides = []

        for rotulo, frases in ancoras_cfg.items():
            vetores_frases = self.gerador.gerar_vetores_em_lote(frases, batch_size=len(frases))
            centroide = np.mean(vetores_frases, axis=0)
            norma = np.linalg.norm(centroide)
            if norma > 0:
                centroide = centroide / norma
            rotulos.append(rotulo)
            vetores_centroides.append(centroide)

        self.rotulos = rotulos
        self.matriz_ancoras = np.array(vetores_centroides, dtype=np.float32)
        self.logger.info(f"Classificador tematico semantico inicializado com {len(self.rotulos)} eixos.")

    def classificar_vetor(self, vetor: np.ndarray) -> str:
        if self.matriz_ancoras is None or len(self.rotulos) == 0:
            return "politico"

        norma = np.linalg.norm(vetor)
        v_norm = vetor / norma if norma > 0 else vetor

        similaridades = np.dot(self.matriz_ancoras, v_norm)
        melhor_idx = int(np.argmax(similaridades))
        return self.rotulos[melhor_idx]

    def classificar_lote_vetores(self, vetores: np.ndarray) -> List[str]:
        if self.matriz_ancoras is None or len(self.rotulos) == 0 or len(vetores) == 0:
            return ["politico"] * len(vetores)

        normas = np.linalg.norm(vetores, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        vetores_norm = vetores / normas

        matriz_sim = np.dot(vetores_norm, self.matriz_ancoras.T)
        indices_max = np.argmax(matriz_sim, axis=1)
        return [self.rotulos[idx] for idx in indices_max]

    def classificar_texto(self, texto: str) -> str:
        vetor = self.gerador.gerar_vetor(texto)
        return self.classificar_vetor(vetor)
