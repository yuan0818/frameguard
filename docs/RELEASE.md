# Experimental v0.1 release

The repository includes source code, reproducible smoke examples, self-contained
tests, MIT license, dependency constraints and GitHub Actions configuration.
See VALIDATION.md for what actually ran. CI configuration is not evidence of a
successful remote run. Full real-world model training and benchmarking remain
outside this source release.

Before distributing weights, train with the corrected source split and retain
the training config and data provenance. Do not present historical validation
metrics as a held-out benchmark. Keep datasets and local environment files out
of Git; .gitignore excludes the existing local artifacts.

A GitHub destination has not been configured by this preparation step.
