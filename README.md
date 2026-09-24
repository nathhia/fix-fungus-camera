# fix-fungus-camera

Remove em lote os defeitos fixos da Canon PowerShot SX170 IS: **fungo** no bloco óptico, uma **mancha marrom** difusa e **pixels quentes** do sensor (os "pontos brancos"). O EXIF original é preservado.

## Uso

```bash
pip install -r requirements.txt

# 1) Gera a máscara (uma vez, ou quando adicionar fotos de calibração/referência)
python fix_fungus.py build-mask --reference reference --calibration calibration
#    -> confira mask/preview.jpg

# 2) Corrige todas as fotos de input/ e grava em output/
python fix_fungus.py process --input input --output output
```

A máscara pronta já está versionada em `mask/`, então o passo 1 só é necessário para regerá-la.

### Pastas de entrada

| Pasta | O que colocar | Para quê |
|---|---|---|
| `calibration/` | Fotos de **campo branco desfocado** (papel perto da lente, parede lisa, céu, TV branca), em vários zooms e aberturas. Podem ser pequenas (ex.: 640×480). | Mapa do fungo por zoom, e como a sombra muda com a abertura. Ver `calibration/README.md`. |
| `reference/` | Algumas fotos normais em **resolução total**. | Resolução da máscara, pixels quentes e intensidade típica da sombra em fotos reais. |
| `input/` | As fotos a corrigir. | — |

Sem `calibration/`, o script tenta estimar o fungo só pelas áreas lisas das fotos de `reference/`. Funciona, mas bem pior.

### Opções de `process`

| Opção | Padrão | O que faz |
|---|---|---|
| `--method` | `hybrid` | `hybrid`, `flatfield` ou `inpaint` (ver abaixo) |
| `--algorithm` | `telea` | `telea` = `cv2.INPAINT_TELEA`, `ns` = `cv2.INPAINT_NS` |
| `--radius` | `5` | `inpaintRadius` do `cv2.inpaint` |
| `--strength` | `auto` | intensidade da correção: `auto` (medida em cada foto) ou um número fixo |
| `--blur` | `auto` | desfoque da sombra: `auto` (medido, guiado pela abertura) ou um número fixo |
| `--max-gain` | `0.4` | teto da correção em log (≈ +49% de brilho), para não "estourar" nenhum ponto |
| `--quality` | `95` | qualidade do JPEG de saída (4:4:4) |
| `--overwrite` | desligado | sobrescreve arquivos já existentes em `output/` |
| `-v` | desligado | mostra, por foto, o zoom, o desfoque e a intensidade usados |

Arquivos que não são imagens válidas, ou com resolução diferente da máscara, são pulados e listados no fim.

## Como funciona

### O defeito

- **O fungo é uma sombra semitransparente.** A cena continua passando por ele, só que escurecida de 1% a 10%. A posição é fixa para cada zoom, mas a sombra fica **mais nítida e forte em F/8** e **mais espalhada e fraca em F/3.5**. Também varia com a luz da cena.
- **A mancha marrom** em ~(2800, 1330), no zoom mínimo, absorve mais azul, então é corrigida por canal de cor.
- **Os pixels quentes** (~20) são do sensor: iguais em qualquer zoom e em todas as fotos.
- **O carimbo de data** da câmera é ignorado, porque é detectado pela cor laranja.

### A correção

1. **Mapa por zoom** (`mask/zoom_*mm/`), feito das fotos de calibração em resolução total e, entre elas, as de maior F/ (sombra mais nítida), junto com uma tabela *abertura → (desfoque, intensidade)* medida nas próprias fotos de calibração.
2. **Em cada foto**, o script lê zoom e abertura no EXIF, prevê o desfoque da sombra e **mede** desfoque e intensidade nas áreas lisas da própria foto. Sem área lisa para medir (folhagem, cena muito texturizada), usa a intensidade da calibração para aquela abertura. A intensidade também varia no espaço: onde a foto não mostra a sombra (textura, objeto na frente), a correção recua em vez de clarear um ponto que não estava escuro.
3. **Correção flat-field:** cada pixel é multiplicado por `exp(k · desfoque(A))`, desfazendo a atenuação e recuperando a textura real sob o filamento. Isso é feito em duas passadas.
   A intensidade `k` é medida **por canal de cor**: a cor da sombra depende da luz da cena (sob céu azul, a mancha marrom rouba bem mais azul do que na calibração feita com luz quente).
   Para medir a sombra, cada pixel é comparado com o **fundo local calculado só entre vizinhos de brilho parecido**, para que o céu claro não faça o mar logo abaixo do horizonte (ou uma parede clara, o móvel escuro ao lado) parecer sombreado.
   **Manchas largas** (a marrom) são tratadas à parte: além de bloquear luz, elas espalham um véu, então escurecem fundo claro e clareiam fundo escuro. Só são corrigidas com evidência medida em regiões de brilho parecido.
   No fim, cada mancha larga ganha um **ajuste final por canal** contra a vizinhança imediata. Sob uma luz diferente da calibração, a mancha marrom pode sobrar (marrom) ou passar do ponto (um ponto claro/azulado), e esse ajuste desfaz o que sobrou nos dois sentidos, recuando quando a medida é incerta (teto de ±8%).
4. **`cv2.inpaint`** (Telea/NS) só nos pixels quentes.

`--method inpaint` aplica só o `cv2.inpaint` na máscara binária. Serve para comparar, mas em filamentos largos ele inventa conteúdo e borra ondas, folhagem e rostos.

### Resultado nas 21 fotos de referência

Medida objetiva da sombra do fungo que continua visível, nas áreas lisas de cada foto (com o mapa de calibração em resolução total e o fundo local que respeita bordas). O processamento leva ~15 s por foto.

| | Antes | Depois |
|---|---|---|
| Céu/mar (3134, 3135, 3151–3155) | 1.0–3.3% | 0.0–0.7% |
| Jantar (3139–3147) | 1.8–3.1% | 0.7–2.3% |
| **Média** | **2.20%** | **0.77%** |

As fotos do jantar em F/3.5 com luz de lâmpada são as mais difíceis. Nelas ainda sobra parte da sombra no alto da parede.

## Preservação de metadados

O bloco EXIF da foto original é copiado **byte a byte** para a saída: data, ISO, abertura, velocidade, distância focal, orientação e o MakerNote da Canon. Ele não é remontado de propósito, porque reserializar o EXIF quebra os offsets internos do MakerNote. A única limitação é que a miniatura embutida (160 px) continua sendo a original.

As fotos são processadas na orientação do **sensor** (ignorando a tag Orientation), porque o defeito é fixo em relação ao sensor. A tag é mantida, então os visualizadores giram a foto normalmente.

## Estrutura

```
fix_fungus.py          atalho para a CLI
fungusfix/
  imageio.py           leitura na orientação do sensor; EXIF/ICC copiados byte a byte
  model.py             modelo da sombra: resíduo por canal, ajuste de desfoque/intensidade
  mask.py              geração da máscara (calibração por zoom, pixels quentes)
  correct.py           flat-field / híbrido / inpaint
  cli.py               build-mask e process (tqdm, tratamento de erros)
mask/                  máscara gerada (mask.png = máscara binária; zoom_*mm/ = mapas por zoom)
calibration/           fotos de campo branco (não versionadas)
reference/             fotos de referência (não versionadas)
```
