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
        self.gerador_embeddings = GeradorEmbeddings()
        self.analisador_sentimento = AnalisadorSentimento()
        self.detector_clickbait = DetectorClickbait()
        self.extrator_entidades = ExtratorEntidades()
        self.calculador_atribuicao = CalculadorAtribuicao()
        self.classificador_tematico = ClassificadorTematicoSemantico()
        self.batch_size = self.config.obter("nlp.batch_size", 64)

    def executar_pre_tratamento(self) -> int:
        removidos = self.banco.limpar_materias_invalidas_ou_curtas(min_tamanho=150)
        if removidos > 0:
            self.logger.info(f"Pre-tratamento NLP: {removidos} materias invalidas ou incompletas removidas.")
        return removidos

    def processar_embeddings(self, callback_progresso=None) -> int:
        materias = self.banco.obter_materias_sem_embedding()
        if not materias:
            return 0

        total_processado = 0
        for i in range(0, len(materias), self.batch_size):
            lote = materias[i:i + self.batch_size]
            textos = [f"{m['titulo']} {m['corpo'][:400]}" for m in lote]
            vetores = self.gerador_embeddings.gerar_vetores_em_lote(textos, batch_size=self.batch_size)
            eixos = self.classificador_tematico.classificar_lote_vetores(vetores)

            dados_banco = []
            dados_eixos = []
            for m, v, eixo in zip(lote, vetores, eixos):
                v_bytes = self.gerador_embeddings.serializar_vetor(v)
                dados_banco.append((m["id"], v_bytes, len(v)))
                dados_eixos.append((eixo, m["id"]))

            self.banco.salvar_embeddings_em_lote(dados_banco)
            self.banco.atualizar_eixos_tematicos_em_lote(dados_eixos)
            total_processado += len(lote)
            if callback_progresso:
                callback_progresso(len(lote))

        self.logger.info(f"Processamento de embeddings e eixos concluido: {total_processado} gerados.")
        return total_processado

    def processar_features_linguisticas(self, callback_progresso=None) -> int:
        materias = self.banco.obter_materias_para_nlp()
        if not materias:
            return 0

        total_processado = 0
        for i in range(0, len(materias), self.batch_size):
            lote = materias[i:i + self.batch_size]
            titulos = [m["titulo"] for m in lote]
            textos_sentimento = [f"{m['titulo']}. {m['corpo'][:300]}" for m in lote]
            textos_entidades = [m["corpo"] for m in lote]

            sentimentos = self.analisador_sentimento.analisar_em_lote(textos_sentimento)
            entidades_lote = self.extrator_entidades.extrair_em_lote(textos_entidades, batch_size=self.batch_size)

            lote_features = []
            lote_entidades = []

            for m, tit, sent, ents in zip(lote, titulos, sentimentos, entidades_lote):
                m_id = m["id"]
                cb_score = self.detector_clickbait.calcular_score(tit)
                lote_features.append((
                    cb_score,
                    sent["polaridade"],
                    sent["pos"],
                    sent["neg"],
                    sent["neu"],
                    m_id
                ))

                if ents:
                    ents_com_atrib = self.calculador_atribuicao.processar_entidades_artigo(m["corpo"], ents)
                    lote_entidades.append((m_id, ents_com_atrib))

            self.banco.salvar_features_em_lote(lote_features)
            self.banco.salvar_entidades_em_lote(lote_entidades)

            total_processado += len(lote)
            if callback_progresso:
                callback_progresso(len(lote))

        self.logger.info(f"Processamento de features linguisticas concluido: {total_processado} processados.")
        return total_processado

    def executar(self, callback_progresso=None) -> Dict[str, int]:
        self.executar_pre_tratamento()
        total_emb = self.processar_embeddings(callback_progresso)
        total_feat = self.processar_features_linguisticas(callback_progresso)
        return {"embeddings": total_emb, "features": total_feat}
