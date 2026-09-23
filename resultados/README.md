# Resultados: antes e depois

As 21 fotos de referência (Canon PowerShot SX170 IS), corrigidas com a versão atual do `fungusfix`.

| Pasta | Conteúdo | Qualidade |
|---|---|---|
| `antes/` | Arquivos originais da câmera | Cópia byte a byte, sem nenhuma recompressão |
| `depois/` | Fotos corrigidas, 4608×3456 | JPEG qualidade 100, cor 4:4:4, EXIF original copiado byte a byte |
| `lado_a_lado/` | Antes \| depois na mesma imagem, resolução total (9240×3456 ou 6936×4608 nas verticais) | JPEG qualidade 95, cor 4:4:4, já girado na orientação de exibição |

Para comparar detalhes, abra `antes/` e `depois/` do mesmo arquivo e alterne entre eles no visualizador:
a imagem inteira fica no mesmo lugar e só o defeito muda. O lado a lado é mais prático para ver a foto toda de uma vez.

Gerado com:

```bash
python fix_fungus.py process --input reference --output resultados/depois --quality 100
```
