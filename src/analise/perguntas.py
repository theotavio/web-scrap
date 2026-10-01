from typing import Any, Dict, List
import numpy as np
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados
from src.analise.estatistica import AnalisadorEstatistico


class RespondedorPerguntas:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("estatisticas")

    def responder_pergunta_principal(self) -> Dict[str, Any]:
        con = self.banco.obter_conexao()
        cur = con.cursor()

        cur.execute("""
            SELECT m.evento_id, m.veiculo_id, v.nome as veiculo_nome,
                   m.sentimento_polaridade, m.score_clickbait
            FROM materias m
            JOIN veiculos v ON m.veiculo_id = v.id
            WHERE m.evento_id > -1
            ORDER BY m.evento_id ASC
        """)
        linhas = [dict(row) for row in cur.fetchall()]

        cur.execute("""
            SELECT em.materia_id, em.texto_entidade, em.polaridade_atribuicao, m.evento_id, m.veiculo_id, v.nome as veiculo_nome
            FROM entidades_mencoes em
            JOIN materias m ON em.materia_id = m.id
            JOIN veiculos v ON m.veiculo_id = v.id
            WHERE m.evento_id > -1
        """)
        linhas_entidades = [dict(row) for row in cur.fetchall()]
        con.close()

        eventos_map: Dict[int, List[Dict[str, Any]]] = {}
        for l in linhas:
            ev_id = l["evento_id"]
            if ev_id not in eventos_map:
                eventos_map[ev_id] = []
            eventos_map[ev_id].append(l)

        eventos_multi_veiculo = {ev_id: mats for ev_id, mats in eventos_map.items() if len(set(m["veiculo_id"] for m in mats)) >= 2}

        divergencias_sentimento = []
        grupos_por_veiculo: Dict[str, List[float]] = {}

        for ev_id, mats in eventos_multi_veiculo.items():
            sents = [m["sentimento_polaridade"] for m in mats]
            desvio = float(np.std(sents))
            divergencias_sentimento.append(desvio)

            for m in mats:
                v_nome = m["veiculo_nome"]
                if v_nome not in grupos_por_veiculo:
                    grupos_por_veiculo[v_nome] = []
                grupos_por_veiculo[v_nome].append(m["sentimento_polaridade"])

        grupos_anova = list(grupos_por_veiculo.values())
        res_anova = AnalisadorEstatistico.teste_anova_kruskal(grupos_anova)
        desc_div = AnalisadorEstatistico.calcular_estatisticas_descritivas(divergencias_sentimento)

        entidades_por_veiculo: Dict[str, List[float]] = {}
        for le in linhas_entidades:
            v_nome = le["veiculo_nome"]
            if v_nome not in entidades_por_veiculo:
                entidades_por_veiculo[v_nome] = []
            entidades_por_veiculo[v_nome].append(le["polaridade_atribuicao"])

        res_anova_entidades = AnalisadorEstatistico.teste_anova_kruskal(list(entidades_por_veiculo.values()))

        resultado = {
            "pergunta": "Pergunta Principal: Divergencia Sistematica de Enquadramento",
            "total_eventos_compartilhados": len(eventos_multi_veiculo),
            "estatisticas_divergencia_sentimento": desc_div,
            "teste_efeito_veiculo_sentimento": res_anova,
            "teste_efeito_veiculo_atribuicao_entidades": res_anova_entidades,
            "conclusao": (
                "Existe divergência sistemática e estatisticamente significativa no enquadramento e sentimento"
                if res_anova.get("significativo")
                else "Os dados indicam convergência factual moderada entre veículos nos mesmos eventos"
            )
        }

        self.banco.salvar_resultado_estatistico(
            pergunta="pergunta_principal",
            chave="divergencia_enquadramento",
            valor=desc_div.get("media", 0.0),
            detalhes=resultado
        )
        return resultado

    def responder_pergunta_clickbait(self) -> Dict[str, Any]:
        con = self.banco.obter_conexao()
        cur = con.cursor()

        cur.execute("""
            SELECT m.score_clickbait, v.tipo as veiculo_tipo, v.nome as veiculo_nome
            FROM materias m
            JOIN veiculos v ON m.veiculo_id = v.id
        """)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()

        tradicionais = [l["score_clickbait"] for l in linhas if l["veiculo_tipo"] == "tradicional"]
        digitais = [l["score_clickbait"] for l in linhas if l["veiculo_tipo"] == "digital_nativo"]

        desc_trad = AnalisadorEstatistico.calcular_estatisticas_descritivas(tradicionais)
        desc_dig = AnalisadorEstatistico.calcular_estatisticas_descritivas(digitais)

        teste_comparativo = AnalisadorEstatistico.teste_diferenca_dois_grupos(digitais, tradicionais)

        veiculo_tipos_map = {l["veiculo_nome"]: l["veiculo_tipo"] for l in linhas}
        ranking_veiculos: Dict[str, List[float]] = {}
        for l in linhas:
            v_nome = l["veiculo_nome"]
            if v_nome not in ranking_veiculos:
                ranking_veiculos[v_nome] = []
            ranking_veiculos[v_nome].append(l["score_clickbait"])

        ranking_desc = {
            v: AnalisadorEstatistico.calcular_estatisticas_descritivas(scores)
            for v, scores in ranking_veiculos.items()
        }
        ranking_ordenado = [
            (v, veiculo_tipos_map.get(v, "-"), stats_dict)
            for v, stats_dict in sorted(ranking_desc.items(), key=lambda x: x[1]["media"], reverse=True)
        ]

        resultado = {
            "pergunta": "Pergunta Secundaria 1: Clickbait e Credibilidade por Tipo de Veiculo",
            "estatisticas_tradicionais": desc_trad,
            "estatisticas_digitais_nativos": desc_dig,
            "teste_hipotese": teste_comparativo,
            "ranking_clickbait_veiculos": ranking_ordenado,
            "conclusao": (
                "Veículos digitais nativos apresentam índices de clickbait significativamente superiores aos veículos tradicionais"
                if teste_comparativo.get("significativo") and desc_dig["media"] > desc_trad["media"]
                else "Não houve diferença estatisticamente expressiva nos índices de clickbait entre tradicionais e digitais nativos"
            )
        }

        self.banco.salvar_resultado_estatistico(
            pergunta="pergunta_clickbait",
            chave="diferenca_tipos_veiculo",
            valor=desc_dig.get("media", 0.0) - desc_trad.get("media", 0.0),
            detalhes=resultado
        )
        return resultado

    def responder_pergunta_ciclos_eleitorais(self) -> Dict[str, Any]:
        con = self.banco.obter_conexao()
        cur = con.cursor()

        cur.execute("""
            SELECT ano, ano_eleitoral, sentimento_polaridade, score_clickbait
            FROM materias
        """)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()

        eleitorais = [abs(l["sentimento_polaridade"]) + (0.5 * l["score_clickbait"]) for l in linhas if l["ano_eleitoral"] == 1]
        nao_eleitorais = [abs(l["sentimento_polaridade"]) + (0.5 * l["score_clickbait"]) for l in linhas if l["ano_eleitoral"] == 0]

        desc_eleit = AnalisadorEstatistico.calcular_estatisticas_descritivas(eleitorais)
        desc_nao_eleit = AnalisadorEstatistico.calcular_estatisticas_descritivas(nao_eleitorais)

        teste_hipotese = AnalisadorEstatistico.teste_diferenca_dois_grupos(eleitorais, nao_eleitorais)

        por_ano: Dict[int, List[float]] = {}
        for l in linhas:
            ano = l["ano"]
            indice_vies = abs(l["sentimento_polaridade"]) + (0.5 * l["score_clickbait"])
            if ano not in por_ano:
                por_ano[ano] = []
            por_ano[ano].append(indice_vies)

        evolucao_anual = {
            ano: AnalisadorEstatistico.calcular_estatisticas_descritivas(scores)
            for ano, scores in sorted(por_ano.items())
        }

        resultado = {
            "pergunta": "Pergunta Secundaria 2: Evolucao Temporal em Ciclos Eleitorais",
            "estatisticas_anos_eleitorais": desc_eleit,
            "estatisticas_anos_nao_eleitorais": desc_nao_eleit,
            "teste_hipotese": teste_hipotese,
            "evolucao_anual": evolucao_anual,
            "conclusao": (
                "O grau de enviesamento textual aumenta de forma estatisticamente perceptível durante anos eleitorais"
                if teste_hipotese.get("significativo") and desc_eleit["media"] > desc_nao_eleit["media"]
                else "A intensidade do viés textual manteve-se estrutural e estável entre anos eleitorais e não eleitorais"
            )
        }

        self.banco.salvar_resultado_estatistico(
            pergunta="pergunta_eleicoes",
            chave="comparativo_eleitoral",
            valor=desc_eleit.get("media", 0.0) - desc_nao_eleit.get("media", 0.0),
            detalhes=resultado
        )
        return resultado

    def responder_pergunta_efeito_mudo(self) -> Dict[str, Any]:
        con = self.banco.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            SELECT m.eixo_tematico, v.tipo as veiculo_tipo, COUNT(m.id) as contagem
            FROM materias m
            JOIN veiculos v ON m.veiculo_id = v.id
            WHERE m.eixo_tematico IS NOT NULL AND m.eixo_tematico != ''
            GROUP BY m.eixo_tematico, v.tipo
        """)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()

        dist_trad: Dict[str, int] = {}
        dist_dig: Dict[str, int] = {}

        for l in linhas:
            eixo = l["eixo_tematico"]
            cnt = l["contagem"]
            if l["veiculo_tipo"] == "tradicional":
                dist_trad[eixo] = cnt
            else:
                dist_dig[eixo] = cnt

        analise_mudo = AnalisadorEstatistico.calcular_efeito_mudo_tematico(dist_trad, dist_dig)

        todos_eixos = sorted(list(set(dist_trad.keys()) | set(dist_dig.keys())))
        tabela_contingencia = [
            [dist_trad.get(e, 0) for e in todos_eixos],
            [dist_dig.get(e, 0) for e in todos_eixos]
        ]
        teste_qui2 = AnalisadorEstatistico.teste_qui_quadrado_eixos(tabela_contingencia)

        resultado = {
            "pergunta": "Pergunta Secundaria 3: Efeito Mudo e Silenciamento Seletivo por Eixo Tematico",
            "distribuicao_tradicional": dist_trad,
            "distribuicao_digital": dist_dig,
            "analise_efeito_mudo": analise_mudo,
            "teste_independencia_qui_quadrado": teste_qui2,
            "conclusao": (
                "Há assimetria expressiva e silenciamento temático diferencial entre a imprensa tradicional e os nativos digitais"
                if teste_qui2.get("significativo")
                else "A distribuição da agenda temática entre os grupos de imprensa manteve proporcionalidade equivalente"
            )
        }

        self.banco.salvar_resultado_estatistico(
            pergunta="pergunta_efeito_mudo",
            chave="divergencia_tematica_global",
            valor=analise_mudo.get("divergencia_efeito_mudo_global", 0.0),
            detalhes=resultado
        )
        return resultado

    def executar_todas(self) -> Dict[str, Any]:
        p1 = self.responder_pergunta_principal()
        p2 = self.responder_pergunta_clickbait()
        p3 = self.responder_pergunta_ciclos_eleitorais()
        p4 = self.responder_pergunta_efeito_mudo()
        return {
            "pergunta_principal": p1,
            "pergunta_clickbait": p2,
            "pergunta_eleicoes": p3,
            "pergunta_efeito_mudo": p4
        }
