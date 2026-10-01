import hashlib
import re
import unicodedata
from typing import Optional


class LimpadorTexto:
    _assinaturas = [
        r"^da reda[çc][ãa]o\s*[-–—:]?",
        r"^folhapress\s*[-–—:]?",
        r"^estad[ãa]o conte[úu]do\s*[-–—:]?",
        r"^ag[êe]ncia brasil\s*[-–—:]?",
        r"^reuters\s*[-–—:]?",
        r"^afp\s*[-–—:]?",
        r"^efe\s*[-–—:]?",
        r"^com ag[êe]ncias\s*[-–—:]?",
        r"^redação\s*[-–—:]?",
        r"^conteúdo estadão\s*[-–—:]?",
        r"publicidade\s*",
        r"veja mais\s*[-–—:]?",
        r"leia tamb[ée]m\s*[-–—:]?",
        r"inscreva-se no canal\s*.*$",
        r"compartilhe no whatsapp\s*",
        r"siga o .* no google notícias",
    ]

    @classmethod
    def normalizar_unicode(cls, texto: str) -> str:
        if not texto:
            return ""
        return unicodedata.normalize("NFKC", texto)

    @classmethod
    def limpar_espacos(cls, texto: str) -> str:
        if not texto:
            return ""
        texto = re.sub(r"\r\n|\r", "\n", texto)
        texto = re.sub(r"[ \t]+", " ", texto)
        texto = re.sub(r"\n\s*\n+", "\n\n", texto)
        return texto.strip()

    @classmethod
    def remover_boilerplate(cls, texto: str) -> str:
        if not texto:
            return ""
        texto_limpo = cls.normalizar_unicode(texto)
        linhas = texto_limpo.split("\n")
        linhas_filtradas = []

        for linha in linhas:
            l_strip = linha.strip()
            if not l_strip:
                continue

            l_baixa = l_strip.lower()
            eh_boilerplate = False

            for padrao in cls._assinaturas:
                if re.search(padrao, l_baixa):
                    l_strip = re.sub(padrao, "", l_strip, flags=re.IGNORECASE).strip()
                    if not l_strip:
                        eh_boilerplate = True
                        break

            if not eh_boilerplate and len(l_strip) > 0:
                linhas_filtradas.append(l_strip)

        resultado = "\n\n".join(linhas_filtradas)
        return cls.limpar_espacos(resultado)

    @classmethod
    def extrair_resumo(cls, texto: str, max_chars: int = 250) -> str:
        if not texto:
            return ""
        paragrafos = [p for p in texto.split("\n\n") if len(p.strip()) > 30]
        if paragrafos:
            primeiro = paragrafos[0].strip()
            if len(primeiro) <= max_chars:
                return primeiro
            return primeiro[:max_chars].rsplit(" ", 1)[0] + "..."
        return texto[:max_chars].rsplit(" ", 1)[0] + "..."

    @classmethod
    def gerar_hash_conteudo(cls, titulo: str, corpo: str) -> str:
        t_norm = cls.normalizar_unicode(titulo).lower().strip()
        c_norm = cls.normalizar_unicode(corpo[:300]).lower().strip()
        base = f"{t_norm}|||{c_norm}".encode("utf-8")
        return hashlib.sha256(base).hexdigest()
