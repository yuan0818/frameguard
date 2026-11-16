# Data and distribution

The MIT license covers this repository's original code. It does not replace
licenses or attribution requirements for downloaded data, pretrained backbones,
or third-party assets. The source repository intentionally excludes datasets
and weights. Obtain data separately from its owner and retain its license and
attributions. Generated demo data is for pipeline checks, not quality claims.

Record the source, version, preprocessing, source group, seed and partition.
Split by original image/recording before augmentation. For related recordings,
provide a common source_id (or group by speaker/session/device as appropriate).
A separate held-out dataset is required for final performance reporting.

Exception CSV columns are `file_name,exception_type` with optional `source_id`.
A normal image's default source ID is its filename stem. Generated corruption
names `<source>_<class>_vNN` automatically reuse that source ID. Supply source_id
for renamed/custom derivatives; use the same ID as the original normal image.
All frames of a video remain in one group. Duplicate basenames conservatively
stay together. Arbitrarily renamed duplicates require explicit provenance.
