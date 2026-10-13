import gc
from typing import Any, Dict, List, Optional
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados
from src.nlp.embeddings import GeradorEmbeddings
from src.nlp.sentimento import AnalisadorSentimento
from src.nlp.clickbait import DetectorClickbait
from src.nlp.entidades import ExtratorEntidades
from src.nlp.atribuicao import CalculadorAtribuicao
from src.nlp.classificador_tematico import ClassificadorTematicoSemantico


class MotorNLP:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("nlp")
        self.batch_size = self.config.obter("nlp.batch_size", 64)
        self.tamanho_lote_banco = self.config.obter("nlp.lote_banco", 500)
        self._gerador_embeddings = None
        self._classificador_tematico = None
        self._analisador_sentimento = None
        self._detector_clickbait = None
        self._extrator_entidades = None
        self._calculador_atribuicao = None

    @property
    def gerador_embeddings(self) -> GeradorEmbeddings:
        if self._gerador_embeddings is None:
            self._gerador_embeddings = GeradorEmbeddings()
        return self._gerador_embeddings

    @property
    def classificador_tematico(self) -> ClassificadorTematicoSemantico:
        if self._classificador_tematico is None:
            self._classificador_tematico = ClassificadorTematicoSemantico()
        return self._classificador_tematico

    @property
    def analisador_sentimento(self) -> AnalisadorSentimento:
        if self._analisador_sentimento is None:
            self._analisador_sentimento = AnalisadorSentimento()
        return self._analisador_sentimento

    @property
    def detector_clickbait(self) -> DetectorClickbait:
        if self._detector_clickbait is None:
            self._detector_clickbait = DetectorClickbait()
        return self._detector_clickbait

    @property
    def extrator_entidades(self) -> ExtratorEntidades:
        if self._extrator_entidades is None:
            self._extrator_entidades = ExtratorEntidades()
        return self._extrator_entidades

    @property
    def calculador_atribuicao(self) -> CalculadorAtribuicao:
        if self._calculador_atribuicao is None:
            self._calculador_atribuicao = CalculadorAtribuicao()
        return self._calculador_atribuicao

    def liberar_modelos_embeddings(self) -> None:
        self._gerador_embeddings = None
        self._classificador_tematico = None
        GeradorEmbeddings._instancia = None
        GeradorEmbeddings._modelo = None
        ClassificadorTematicoSemantico._instancia = None
        ClassificadorTematicoSemantico._ancoras_carregadas = False
        gc.collect()

    def liberar_modelos_linguisticos(self) -> None:
        self._analisador_sentimento = None
        self._extrator_entidades = None
        self._detector_clickbait = None
        self._calculador_atribuicao = None
        AnalisadorSentimento._instancia = None
        AnalisadorSentimento._analisador = None
        ExtratorEntidades._instancia = None
        ExtratorEntidades._nlp = None
        gc.collect()

    def executar_pre_tratamento(self) -> int:
        removidos = self.banco.limpar_materias_invalidas_ou_curtas(min_tamanho=150)
        if removidos > 0:
            self.logger.info(f"Pre-tratamento NLP: {removidos} materias invalidas ou incompletas removidas.")
        return removidos

    def processar_embeddings(self, limite: Optional[int] = None, veiculo_id: Optional[int] = None, callback_progresso=None) -> int:
        total_processado = 0
        while True:
            qtd_buscar = self.tamanho_lote_banco
            if limite is not None:
                restante = limite - total_processado
                if restante <= 0:
                    break
                qtd_buscar = min(self.tamanho_lote_banco, restante)

            materias = self.banco.obter_materias_sem_embedding(limite=qtd_buscar, veiculo_id=veiculo_id)
            if not materias:
                break

            qtd_obtida = len(materias)
            for i in range(0, qtd_obtida, self.batch_size):
                sub_lote = materias[i:i + self.batch_size]
                textos = [f"{m['titulo']} {m['corpo'][:400]}" for m in sub_lote]
                vetores = self.gerador_embeddings.gerar_vetores_em_lote(textos, batch_size=self.batch_size)
                eixos = self.classificador_tematico.classificar_lote_vetores(vetores)

                dados_banco = []
                dados_eixos = []
                for m, v, eixo in zip(sub_lote, vetores, eixos):
                    v_bytes = self.gerador_embeddings.serializar_vetor(v)
                    dados_banco.append((m["id"], v_bytes, len(v)))
                    dados_eixos.append((eixo, m["id"]))

                self.banco.salvar_embeddings_em_lote(dados_banco)
                self.banco.atualizar_eixos_tematicos_em_lote(dados_eixos)
                total_processado += len(sub_lote)
                if callback_progresso:
                    callback_progresso(len(sub_lote))

            del materias
            gc.collect()

            if qtd_obtida < qtd_buscar:
                break

        self.logger.info(f"Processamento de embeddings e eixos concluido: {total_processado} gerados.")
        return total_processado

    def processar_features_linguisticas(self, limite: Optional[int] = None, veiculo_id: Optional[int] = None, callback_progresso=None) -> int:
        total_processado = 0
        tamanho_lote = min(self.tamanho_lote_banco, 200)
        while True:
            qtd_buscar = tamanho_lote
            if limite is not None:
                restante = limite - total_processado
                if restante <= 0:
                    break
                qtd_buscar = min(tamanho_lote, restante)

            materias = self.banco.obter_materias_para_nlp(limite=qtd_buscar, veiculo_id=veiculo_id)
            if not materias:
                break

            qtd_obtida = len(materias)
            for i in range(0, qtd_obtida, self.batch_size):
                sub_lote = materias[i:i + self.batch_size]
                titulos = [m["titulo"] for m in sub_lote]
                textos_sentimento = [f"{m['titulo']}. {m['corpo'][:300]}" for m in sub_lote]
                textos_entidades = [m["corpo"] for m in sub_lote]

                sentimentos = self.analisador_sentimento.analisar_em_lote(textos_sentimento)
                entidades_lote = self.extrator_entidades.extrair_em_lote(textos_entidades, batch_size=self.batch_size)

                lote_features = []
                lote_entidades = []

                for m, tit, sent, ents in zip(sub_lote, titulos, sentimentos, entidades_lote):
                    m_id = m["id"]
                    cb_score = self.detector_clickbait.calcular_score(tit)
                    neu_score = sent["neu"]
                    if neu_score == 0.0 and sent["polaridade"] == 0.0 and cb_score == 0.0:
                        neu_score = 0.0001
                    lote_features.append((
                        cb_score,
                        sent["polaridade"],
                        sent["pos"],
                        sent["neg"],
                        neu_score,
                        m_id
                    ))

                    if ents:
                        ents_com_atrib = self.calculador_atribuicao.processar_entidades_artigo(m["corpo"], ents)
                        lote_entidades.append((m_id, ents_com_atrib))

                self.banco.salvar_features_em_lote(lote_features)
                self.banco.salvar_entidades_em_lote(lote_entidades)

                total_processado += len(sub_lote)
                if callback_progresso:
                    callback_progresso(len(sub_lote))

            del materias
            gc.collect()

            if qtd_obtida < qtd_buscar:
                break

        self.logger.info(f"Processamento de features linguisticas concluido: {total_processado} processados.")
        return total_processado

    def executar(self, limite: Optional[int] = None, veiculo_id: Optional[int] = None, callback_progresso=None) -> Dict[str, int]:
        total_emb = self.processar_embeddings(limite=limite, veiculo_id=veiculo_id, callback_progresso=callback_progresso)
        self.liberar_modelos_embeddings()
        total_feat = self.processar_features_linguisticas(limite=limite, veiculo_id=veiculo_id, callback_progresso=callback_progresso)
        self.liberar_modelos_linguisticos()
        return {"embeddings": total_emb, "features": total_feat}
