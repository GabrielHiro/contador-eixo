# Testes

Os testes e fixtures ficam separados do código de produção.

- `fixtures/`: imagens e entradas pequenas usadas nos testes.
- `artifacts/`: saídas temporárias de validação; não versionar resultados grandes.
- `smoke_test.sh`: valida sintaxe Python, build C++ e endpoints quando os serviços estão ativos.

Executar a validação:

```bash
bash tests/smoke_test.sh
```

O script não inicia, reinicia ou altera os serviços. Para uma validação completa, suba o backend
na porta 8080 e a interface na porta 8090 antes de executá-lo.
