# calibration/

Fotos de **campo branco desfocado** (papel a poucos cm da lente, parede lisa, céu limpo, TV branca),
que ocupem o quadro inteiro. Elas mostram só o defeito da lente, sem cena, e são a base do mapa do fungo.

- Em **cada posição de zoom** que você usa (o script cria um mapa por zoom a partir do EXIF).
- No zoom mínimo, em várias aberturas (ex.: F/3.5, F/5.6, F/8): o script mede como a sombra
  muda com a abertura e usa isso para corrigir cada foto.
- 3 fotos por combinação, sem estourar o branco.
- Pode ser em resolução menor (ex.: 640x480); as fotos em `reference/` definem a resolução final.

Não são versionadas; a máscara gerada a partir delas fica em `mask/`.
