# Security

## Trusted Model Files

AeroSurrogate models use Python pickle serialization. Loading an untrusted
pickle can execute arbitrary code.

- use the model bundled with the installed package, or
- verify an external model against the SHA-256 checksum in its deployment
  manifest before loading it

Never load model files received from an untrusted source.

## Reporting A Vulnerability

Please report security concerns privately to the project author before opening
a public issue containing exploit details.
