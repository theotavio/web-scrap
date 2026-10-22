import gc
import os
from typing import Dict, List, Optional
import torch
from src.configuracao import obter_configuracao
from src.registro import obter_logger


class AnalisadorSentimento:
    _instancia = None
    _sessao_ort = None
    _modelo_torch = None
    _tokenizer = None
    _analisador = None
    _usando_onnx = False

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(AnalisadorSentimento, cls).__new__(cls)
        return cls._instancia

    def __init__(self):
        if self._sessao_ort is None and self._modelo_torch is None and self._analisador is None:
            self.logger = obter_logger("nlp")
            self.config = obter_configuracao()
            motor = self.config.obter("nlp.motor_sentimento", "onnx_int8")
            self._inicializar_motor(motor)

    def _inicializar_motor(self, motor: str) -> None:
        if motor in ("onnx_int8", "onnx"):
            sucesso_onnx = self._inicializar_onnx(motor)
            if sucesso_onnx:
                return

        sucesso_torch = self._inicializar_torch()
        if sucesso_torch:
            return

        self._inicializar_legado()

    def _inicializar_onnx(self, motor: str) -> bool:
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            caminho_modelo = "dados/modelos/roberta_sentiment_int8.onnx"
            if not os.path.exists(caminho_modelo):
                sucesso_geracao = self._garantir_modelo_onnx(caminho_modelo)
                if not sucesso_geracao or not os.path.exists(caminho_modelo):
                    return False

            sess_options = ort.SessionOptions()
            total_cpus = os.cpu_count() or 4
            sess_options.intra_op_num_threads = min(8, max(1, total_cpus))
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            AnalisadorSentimento._sessao_ort = ort.InferenceSession(
                caminho_modelo,
                sess_options,
                providers=["CPUExecutionProvider"]
            )
            AnalisadorSentimento._tokenizer = AutoTokenizer.from_pretrained(
                "pysentimiento/bertweet-pt-sentiment"
            )
            AnalisadorSentimento._usando_onnx = True
            self.logger.info("Motor de sentimento ONNX Runtime INT8 carregado com sucesso.")
            return True
        except Exception as e:
            self.logger.warning(f"Nao foi possivel carregar motor ONNX Runtime: {e}. Alternando para PyTorch.")
            return False

    def _garantir_modelo_onnx(self, caminho_onnx_int8: str) -> bool:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            pasta = os.path.dirname(caminho_onnx_int8)
            os.makedirs(pasta, exist_ok=True)
            caminho_temp_fp32 = os.path.join(pasta, "roberta_temp.onnx")

            self.logger.info("Exportando modelo RoBERTa para formato ONNX...")
            tokenizer = AutoTokenizer.from_pretrained("pysentimiento/bertweet-pt-sentiment")
            modelo = AutoModelForSequenceClassification.from_pretrained("pysentimiento/bertweet-pt-sentiment")
            modelo.eval()

            entrada_falsa = tokenizer(["Texto base para exportacao de grafo."], padding="max_length", max_length=128, return_tensors="pt")
            torch.onnx.export(
                modelo,
                (entrada_falsa["input_ids"], entrada_falsa["attention_mask"]),
                caminho_temp_fp32,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size"}
                },
                opset_version=14,
                dynamo=False
            )

            self.logger.info("Quantizando modelo ONNX para INT8...")
            quantize_dynamic(
                caminho_temp_fp32,
                caminho_onnx_int8,
                weight_type=QuantType.QInt8
            )

            if os.path.exists(caminho_temp_fp32):
                os.remove(caminho_temp_fp32)

            del modelo
            del tokenizer
            gc.collect()
            return True
        except Exception as e:
            self.logger.error(f"Erro na geracao do modelo ONNX INT8: {e}")
            return False

    def _inicializar_torch(self) -> bool:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            AnalisadorSentimento._tokenizer = AutoTokenizer.from_pretrained(
                "pysentimiento/bertweet-pt-sentiment"
            )
            modelo = AutoModelForSequenceClassification.from_pretrained(
                "pysentimiento/bertweet-pt-sentiment"
            )
            modelo.eval()
            AnalisadorSentimento._modelo_torch = modelo
            AnalisadorSentimento._usando_onnx = False
            self.logger.info("Motor de sentimento PyTorch nativo (inference_mode) carregado com sucesso.")
            return True
        except Exception as e:
            self.logger.warning(f"Nao foi possivel carregar motor PyTorch nativo: {e}")
            return False

    def _inicializar_legado(self) -> None:
        try:
            from pysentimiento import create_analyzer
            AnalisadorSentimento._analisador = create_analyzer(task="sentiment", lang="pt")
            self.logger.info("Motor de sentimento pysentimiento (legado) carregado.")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar motor legado pysentimiento: {e}")
            AnalisadorSentimento._analisador = None

    @classmethod
    def liberar(cls) -> None:
        cls._instancia = None
        cls._sessao_ort = None
        cls._modelo_torch = None
        cls._tokenizer = None
        cls._analisador = None
        cls._usando_onnx = False
        gc.collect()

    def analisar_texto(self, texto: str) -> Dict[str, float]:
        if not texto or not texto.strip():
            return {"polaridade": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        resultados = self.analisar_em_lote([texto])
        if resultados:
            return resultados[0]
        return self._analisar_heuristico(texto[:512])

    def analisar_em_lote(self, textos: List[str]) -> List[Dict[str, float]]:
        if not textos:
            return []

        textos_truncados = [t[:512] if t else "" for t in textos]
        max_tokens = min(getattr(self._tokenizer, "model_max_length", 128) or 128, 128)

        if self._sessao_ort is not None and self._tokenizer is not None:
            try:
                inputs = self._tokenizer(
                    textos_truncados,
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="np"
                )
                ort_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
                ort_outs = self._sessao_ort.run(None, ort_inputs)
                with torch.inference_mode():
                    probs = torch.softmax(torch.from_numpy(ort_outs[0]), dim=-1).tolist()

                resultados = []
                for p in probs:
                    neg = float(p[0])
                    neu = float(p[1])
                    pos = float(p[2])
                    polaridade = round(pos - neg, 4)
                    resultados.append({
                        "polaridade": polaridade,
                        "pos": round(pos, 4),
                        "neg": round(neg, 4),
                        "neu": round(neu, 4)
                    })
                return resultados
            except Exception as e:
                self.logger.error(f"Erro na predicao ONNX de sentimento em lote: {e}")

        if self._modelo_torch is not None and self._tokenizer is not None:
            try:
                with torch.inference_mode():
                    inputs = self._tokenizer(
                        textos_truncados,
                        padding=True,
                        truncation=True,
                        max_length=max_tokens,
                        return_tensors="pt"
                    )
                    outputs = self._modelo_torch(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).tolist()

                resultados = []
                for p in probs:
                    neg = float(p[0])
                    neu = float(p[1])
                    pos = float(p[2])
                    polaridade = round(pos - neg, 4)
                    resultados.append({
                        "polaridade": polaridade,
                        "pos": round(pos, 4),
                        "neg": round(neg, 4),
                        "neu": round(neu, 4)
                    })
                return resultados
            except Exception as e:
                self.logger.error(f"Erro na predicao PyTorch nativo de sentimento em lote: {e}")

        if self._analisador is not None:
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
                self.logger.error(f"Erro no processamento em lote legado de sentimento: {e}")

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
