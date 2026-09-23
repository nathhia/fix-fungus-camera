# fix-fungus-camera

Remove em lote os artefatos fixos da Canon PowerShot SX170 IS: **fungo** no bloco óptico e **pixels quentes** do sensor (o "ponto branco"). O EXIF original é preservado.

## Uso rápido

```bash
pip install -r requirements.txt

# 1) Gera a máscara uma única vez, a partir das fotos em reference/
python fix_fungus.py build-mask --reference reference --mask-dir mask
#    -> confira mask/preview.jpg (fungo em magenta, pixels quentes circulados)

# 2) Corrige todas as fotos de input/ e grava em output/
python fix_fungus.py process --input input --output output
```

A máscara pronta já está versionada em `mask/`, então o passo 1 só é necessário para regerá-la (por exemplo, depois de adicionar mais referências).

### Opções de `process`

| Opção | Padrão | O que faz |
|---|---|---|
| `--method` | `hybrid` | `hybrid`, `flatfield` ou `inpaint` (ver abaixo) |
| `--algorithm` | `telea` | `telea` = `cv2.INPAINT_TELEA`, `ns` = `cv2.INPAINT_NS` |
| `--radius` | `5` | `inpaintRadius` do `cv2.inpaint` |
| `--strength` | `auto` | intensidade do flat-field: `auto` (estimada por foto) ou um número (ex. `1.2`) |
| `--quality` | `95` | qualidade do JPEG de saída (subamostragem 4:4:4) |
| `--overwrite` | desligado | sobrescreve arquivos que já existem em `output/` |

Exemplo usando só inpainting, com Navier-Stokes e raio 3:

```bash
python fix_fungus.py process --method inpaint --algorithm ns --radius 3
```

Arquivos que não são imagens válidas, ou que têm resolução diferente da máscara, são pulados e listados no fim.

## O que a análise das fotos mostrou

1. **O fungo é uma sombra semitransparente, não um objeto opaco.** Os filamentos escurecem a cena em 2–10% e são largos e desfocados (dezenas de px), cobrindo boa parte do terço superior e do centro do quadro. A luz da cena continua passando por eles.
2. **Os "pontos brancos" são pixels quentes do sensor**: cerca de 20 pontos de 1–3 px, alguns brancos e outros avermelhados. Aparecem em todas as fotos, em qualquer zoom.
3. **O padrão é fixo nas coordenadas do sensor.** Por isso o script lê os pixels *sem* aplicar a rotação do EXIF (fotos em retrato têm `Orientation=8`). Aplicar a máscara na imagem girada a desalinharia.
4. **Com o zoom, a sombra muda um pouco.** Em 5 mm ela fica mais nítida e se desloca até cerca de 20 px em relação a 26–37 mm. Por isso existe um mapa por faixa de zoom (`mask/zoom_5.0mm/`), escolhido pela distância focal no EXIF.
5. **A câmera imprime a data na foto em uma posição fixa.** O detector de pixels quentes descarta esse carimbo porque os candidatos formam um aglomerado, e pixel quente é isolado.

### Como a máscara é gerada (`fungusfix/mask.py`)

- **Fungo:** em cada referência, calcula `log(imagem) − log(fundo local)` só nas regiões lisas (céu, parede) e faz a **mediana entre todas as fotos**. O conteúdo das cenas se cancela e sobra o padrão fixo. A binarização usa **limiar por histerese**: uma semente forte (0.02) mais crescimento conectado (0.008), que é o que segue filamentos longos e fracos sem pegar ruído solto.
- **Pixels quentes:** um pixel precisa sobressair da mediana 7×7 em pelo menos 70% das referências, e o candidato precisa ser pequeno e isolado.

### Por que o método padrão é `hybrid` e não só `inpaint`

`cv2.inpaint` *inventa* o conteúdo sob a máscara a partir das bordas. Isso funciona bem para defeitos pequenos e opacos, como os pixels quentes. Nos filamentos largos, que cobrem boa parte do quadro, o inpaint borra ondas, folhagem e rostos e deixa manchas no céu.

Como o fungo só atenua a luz, dá para **desfazer a atenuação** (*flat-field*): multiplicar cada pixel por `exp(A)`, onde `A` é a atenuação medida, recupera a textura original sob o filamento. A intensidade é ajustada por foto (`--strength auto`), porque abertura e zoom mudam o quanto a sombra escurece.

O modo `hybrid` combina os dois: flat-field no fungo e `cv2.inpaint` (Telea/NS, com o raio escolhido) nos pixels quentes e nos núcleos quase opacos do fungo.

## Para melhorar a máscara

Hoje as referências lisas cobrem bem a metade superior do quadro e mal a inferior (areia e folhagem). O que mais melhoraria o resultado:

- Fotografar **um fundo liso e claro** (parede branca, céu sem nuvens, folha de papel bem perto da lente para ficar desfocada) **preenchendo o quadro inteiro**.
- Repetir **em cada posição de zoom que você usa**, principalmente a grande angular (5 mm), e em 2–3 aberturas.
- Colocar essas fotos em `reference/` e rodar `build-mask` de novo.

## Estrutura

```
fix_fungus.py          atalho para a CLI
fungusfix/
  imageio.py           leitura na orientação do sensor, escrita com EXIF/ICC copiados byte a byte
  mask.py              geração da máscara (fungo + pixels quentes, grupos de zoom)
  correct.py           inpaint / flat-field / híbrido
  cli.py               build-mask e process (tqdm, tratamento de erros)
mask/                  máscara gerada (mask.png = máscara binária final)
reference/             suas fotos de referência (não versionadas)
```

## Preservação de metadados

O bloco EXIF da foto original é copiado **byte a byte** para a saída, incluindo data, ISO, abertura, velocidade, distância focal, orientação e o MakerNote da Canon. Ele não é remontado de propósito: reserializar o EXIF quebra os offsets internos do MakerNote. A única limitação é que a miniatura embutida (160 px), que alguns visualizadores usam como prévia, continua sendo a original.
