from typing import Any, Dict, List, Tuple
import numpy as np
import scipy.stats as stats


class AnalisadorEstatistico:
    @staticmethod
    def calcular_estatisticas_descritivas(valores: List[float]) -> Dict[str, float]:
        if not valores:
            return {"media": 0.0, "mediana": 0.0, "desvio_padrao": 0.0, "minimo": 0.0, "maximo": 0.0, "contagem": 0}
        arr = np.array(valores, dtype=float)
        return {
            "media": round(float(np.mean(arr)), 4),
            "mediana": round(float(np.median(arr)), 4),
            "desvio_padrao": round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, 4),
            "minimo": round(float(np.min(arr)), 4),
            "maximo": round(float(np.max(arr)), 4),
            "contagem": int(len(arr))
        }

    @staticmethod
    def teste_diferenca_dois_grupos(grupo1: List[float], grupo2: List[float]) -> Dict[str, Any]:
        if len(grupo1) < 2 or len(grupo2) < 2:
            return {
                "t_stat": 0.0,
                "t_pvalor": 1.0,
                "mannwhitney_u": 0.0,
                "mannwhitney_pvalor": 1.0,
                "d_cohen": 0.0,
                "significativo": False
            }

        g1 = np.array(grupo1, dtype=float)
        g2 = np.array(grupo2, dtype=float)

        t_res = stats.ttest_ind(g1, g2, equal_var=False)
        mw_res = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        n1, n2 = len(g1), len(g2)
        s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1, (n1 + n2 - 2)))
        d_cohen = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0.0

        return {
            "t_stat": round(float(t_res.statistic), 4),
            "t_pvalor": round(float(t_res.pvalue), 6),
            "mannwhitney_u": round(float(mw_res.statistic), 4),
            "mannwhitney_pvalor": round(float(mw_res.pvalue), 6),
            "d_cohen": round(float(d_cohen), 4),
            "significativo": bool(mw_res.pvalue < 0.05)
        }

    @staticmethod
    def teste_anova_kruskal(grupos: List[List[float]]) -> Dict[str, Any]:
        grupos_validos = [g for g in grupos if len(g) >= 2]
        if len(grupos_validos) < 2:
            return {
                "f_stat": 0.0,
                "f_pvalor": 1.0,
                "kruskal_h": 0.0,
                "kruskal_pvalor": 1.0,
                "significativo": False
            }

        f_res = stats.f_oneway(*grupos_validos)
        kw_res = stats.kruskal(*grupos_validos)

        return {
            "f_stat": round(float(f_res.statistic), 4),
            "f_pvalor": round(float(f_res.pvalue), 6),
            "kruskal_h": round(float(kw_res.statistic), 4),
            "kruskal_pvalor": round(float(kw_res.pvalue), 6),
            "significativo": bool(kw_res.pvalue < 0.05)
        }

    @staticmethod
    def teste_qui_quadrado_eixos(tabela_contingencia: List[List[int]]) -> Dict[str, Any]:
        arr = np.array(tabela_contingencia)
        if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2 or np.sum(arr) == 0:
            return {
                "chi2_stat": 0.0,
                "p_valor": 1.0,
                "graus_liberdade": 0,
                "significativo": False
            }

        try:
            chi2, p_val, dof, _ = stats.chi2_contingency(arr)
            return {
                "chi2_stat": round(float(chi2), 4),
                "p_valor": round(float(p_val), 6),
                "graus_liberdade": int(dof),
                "significativo": bool(p_val < 0.05)
            }
        except Exception:
            return {
                "chi2_stat": 0.0,
                "p_valor": 1.0,
                "graus_liberdade": 0,
                "significativo": False
            }

    @staticmethod
    def calcular_efeito_mudo_tematico(dist_tradicional: Dict[str, int], dist_digital: Dict[str, int]) -> Dict[str, Any]:
        todos_eixos = sorted(list(set(dist_tradicional.keys()) | set(dist_digital.keys())))
        total_trad = max(1, sum(dist_tradicional.values()))
        total_dig = max(1, sum(dist_digital.values()))

        detalhes_eixos = {}
        divergencia_total = 0.0

        for eixo in todos_eixos:
            c_trad = dist_tradicional.get(eixo, 0)
            c_dig = dist_digital.get(eixo, 0)

            p_trad = c_trad / total_trad
            p_dig = c_dig / total_dig

            razao = round(p_trad / p_dig, 3) if p_dig > 0 else 999.0
            diff = abs(p_trad - p_dig)
            score_mudo = diff * np.log2(1.0 + (diff / (min(p_trad, p_dig) + 0.001)))

            divergencia_total += float(score_mudo)
            detalhes_eixos[eixo] = {
                "contagem_tradicional": c_trad,
                "proporcao_tradicional": round(p_trad, 4),
                "contagem_digital": c_dig,
                "proporcao_digital": round(p_dig, 4),
                "razao_cobertura_trad_dig": razao,
                "indice_efeito_mudo": round(float(score_mudo), 4)
            }

        return {
            "divergencia_efeito_mudo_global": round(float(divergencia_total), 4),
            "eixos": detalhes_eixos
        }
