# Standard-library compression codecs

gzip, bzip2 and LZMA at three levels, over reproducible text, binary and mixed
corpora. Both directions are timed, so the cost of compressing can be weighed
against the cost of reading the result back.

`round_trip_ok` is 1 when decompression reproduced the input byte for byte;
anything else means the rest of that record is meaningless.

Corpora are generated in point setup over `(kind, mebibytes)` only, so a corpus
is built once and reused by all nine codec/level combinations. Each seed is
derived from the corpus identity, so generation order and parallelism cannot
change the bytes.

```sh
yuclid run -p quick -o yuclid.results.jsonl

yuclid tplot yuclid.results.jsonl -x level -z codec -y compression_seconds -f kind=text
yuclid tplot yuclid.results.jsonl -x codec -z kind -y ratio -f level=6
```

The `larger` preset builds 8 and 32 MiB corpora, where the memory differences
between the codecs become visible.
