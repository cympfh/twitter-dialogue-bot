# twitter dialogue bot

based on Encoder-Decoder model with LSTMs.

## learning plan

1. autoencoder
    - `Encoder -> Decoder` for `(x, x)`
2. encoder-decoder
    - `Encoder -> Decoder` for `(x, y)`
3. encode-to-encode
    - `{Encoder, Encoder} -> f -> 0` for (x1, x2)
