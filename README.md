# workshop-agentes

Material do workshop **"Do Chatbot ao Agente: Construindo Sistemas de IA com Ferramentas"**.

Um projeto mínimo e modular que mostra a evolução de uma aplicação com LLM para um **agente** que usa ferramentas e
raciocina em passos, mantendo o contexto até entregar o resultado:

- **modelo** (Groq via `langchain-groq`), que só *pede* as chamadas de ferramenta; quem executa é o nosso código;
- **memória de trabalho**: a API é *stateless*, então o histórico completo é reenviado a cada passo;
- **registry de ferramentas**: `update_plan` (raciocínio explícito; a 1ª chamada é obrigada a usá-la); a resposta final
  vem em texto (`finish`, uma ferramenta terminal, é opcional);
- **loop manual** de ferramentas, sem agentes prontos, para enxergar cada passo;
- **tracing** próprio: tokens, latência, `finish_reason`, tentativas retentadas, plano por passo e motivo de parada;
- **RAG** como ferramenta (`search_documents`): busca local nos PDFs de `documentos/` (o exemplo é o PPC 2024 de
  Ciência da Computação, `documentos/PPC2024BCC.pdf`);
- **evals**: golden set, checks determinísticos lidos do tracing e juiz LLM opcional.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedroop32/workshop-agentes/blob/main/workshop.ipynb)

## Estrutura

```
workshop_agentes/
  llm.py              build_llm, invoke_with_retry (único ponto de retry), error_label, text_of, list_models
  rag.py              RAG sobre PDFs (PdfRag: busca híbrida) e o chat do executar_rag.sh
  agent/config.py     DEFAULT_MODEL, get_secret, AgentConfig
  agent/memory.py     WorkingMemory (histórico + invariante chamada -> resultado)
  agent/loop.py       Agent e AgentResult (o loop)
  tools/registry.py   ToolRegistry (executa ferramentas sem nunca levantar exceção)
  tools/core.py       update_plan e finish (terminal opcional)
  tools/documents.py  search_documents: o RAG como ferramenta do agente
  tracing/tracer.py   Span e Tracer
  tracing/report.py   relatório em texto
  evals/              golden set, checks, juiz LLM opcional, runner, relatório e CLI (python -m workshop_agentes.evals)
tests/                testes offline (pytest) com modelo falso
  visual.py           visualizações do notebook (HTML): resultado, ferramentas, tracing, memória e evals
workshop.ipynb        notebook do workshop (Colab)
```

## Chave da API: uma conta Groq por aluno

1. Crie uma conta e a sua chave no [console do Groq](https://console.groq.com/keys) (plano gratuito).
2. **Cada aluno usa a própria conta.** Os limites valem **por organização**, não por chave: chaves da mesma
   organização dividem a cota. Os limites da sua conta aparecem na página *Limits* do console
   (referência: [rate limits](https://console.groq.com/docs/rate-limits)).
3. **Nunca** cole a chave numa célula, num arquivo versionado ou numa mensagem.

## Rodando no Colab

1. Abra o notebook pelo botão acima.
2. No painel **Secrets** (ícone de chave 🔑 na barra lateral), crie o secret `GROQ_API_KEY` com a sua chave.
   Opcional: `GROQ_MODEL` para trocar o modelo padrão.
3. **Ative a chave "Acesso ao notebook"** desse secret. Sem isso o notebook não consegue ler a chave.
4. Rode as seções em ordem: ambiente, conexão, montar o agente, agente sem ferramentas, a ferramenta certa, RAG,
   bastidores (tracing e memória), avaliação. A seção 9 (limitações) é só texto.

Se o repositório for atualizado depois de você importar o pacote, reinicie o ambiente de execução
(*Ambiente de execução → Reiniciar sessão*) e rode as células de novo.

## Rodando localmente

Requer Python 3.10+. O `requirements.txt` fixa faixas compatíveis das dependências; versões testadas (2026-10-22):
langchain-core 1.6.12, langchain-groq 1.1.3 (com groq 0.37.1) e python-dotenv 1.2.4.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env      # preencha GROQ_API_KEY (o .env não é versionado)
```

Exemplo mínimo (o mesmo que o notebook faz):

```python
from workshop_agentes.agent.config import DEFAULT_MODEL, AgentConfig, get_secret
from workshop_agentes.agent.loop import Agent
from workshop_agentes.agent.memory import WorkingMemory
from workshop_agentes.llm import build_llm
from workshop_agentes.tools.core import update_plan
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.report import format_report
from workshop_agentes.tracing.tracer import Tracer

config = AgentConfig()
llm = build_llm(get_secret("GROQ_MODEL", DEFAULT_MODEL), get_secret("GROQ_API_KEY"), config.temperature)
registry = ToolRegistry()
registry.register(update_plan)  # resposta final em texto; finish é opcional
agent = Agent(llm, registry, WorkingMemory(config.system_prompt), Tracer(), config)

result = agent.run("Compare lista encadeada e array dinâmico para implementar uma fila e conclua qual usar.")
print(result.reason, result.finish_reason, result.answer)
print(format_report(result.spans))
```

## RAG: perguntas sobre seus PDFs

A pasta para colocar documentos é **`documentos/`**, na raiz do projeto.
No Colab: `/content/workshop-agentes/documentos/`.
Copie os PDFs para ela; subpastas também funcionam. A busca pode consultar todos
os arquivos ou somente um PDF escolhido.

```bash
.venv/bin/python -m pip install -r requirements-rag.txt
./executar_rag.sh
```

O script usa o Python de `.venv`, lê `GROQ_API_KEY` e `GROQ_MODEL` do `.env`
existente e abre um chat no terminal. O primeiro uso baixa o modelo de busca
multilíngue (cerca de 120 MB, mais arquivos auxiliares); os demais reutilizam
o cache em `.rag/`. Não é necessário banco de dados nem outra chave de API.

```bash
./executar_rag.sh --listar
./executar_rag.sh --indexar
./executar_rag.sh "Qual é o prazo informado nos documentos?"
./executar_rag.sh --pdf "meu_arquivo.pdf" "Qual é o objetivo deste documento?"
./executar_rag.sh --pdf contrato_a.pdf --pdf contrato_b.pdf "Compare os prazos dos dois contratos."
```

Para ler **dois (ou mais) PDFs ao mesmo tempo**, repita `--pdf`. Cada PDF escolhido
recebe sua própria cota de trechos (4 cada, com dois PDFs), intercalados, para que um
documento grande não tire o espaço do outro. A resposta diz a qual arquivo pertence
cada informação, e uma citação nunca é atribuída a um PDF diferente do que a contém.

No chat, `/listar` mostra os arquivos (com `*` nos escolhidos), `/pdf a.pdf b.pdf`
escolhe os PDFs consultados, `/pdf` sozinho volta a consultar todos e `/sair` encerra.
Use aspas em nomes com espaços: `/pdf "meu arquivo.pdf" outro.pdf`. Cada pergunta é
independente: escreva a pergunta completa. Novos PDFs, alterações e remoções são
detectados antes da próxima pergunta. As páginas citadas são as posições no PDF,
começando em 1, e podem diferir da numeração impressa no documento.

O sistema combina embeddings locais e BM25, recupera até seis trechos, gera
afirmações com citações literais e valida os trechos no código. Uma segunda
chamada ao modelo revisa cada parágrafo separadamente. Parágrafos sem evidência
válida ou reprovados na revisão são retirados; os demais compõem a resposta.
Aceita paráfrases e respostas parciais úteis quando o contexto não cobre toda a
pergunta. Recusa quando nenhum parágrafo tem suporte. Essa revisão também pode errar.
Não há uma nota mínima de 0.7 ou 0.8 que bloqueie as respostas: a nota ao final
é apenas um indicador de cobertura textual. A geração e a revisão têm limites
de 1400 e 600 tokens, respectivamente, para reduzir a demanda por saída na API.
Trechos recuperados são enviados ao Groq; os PDFs completos e o índice ficam
locais e são ignorados pelo Git (exceto o PDF de exemplo do workshop, `documentos/PPC2024BCC.pdf`). Em geral são duas chamadas por resposta, além
de eventuais retries, consumindo a cota Groq.

Antes de indexar, cabeçalhos e rodapés repetidos (como o nome da revista com o número
da página) são removidos, palavras partidas no fim da linha (`forma-` + `ção`) são
unidas e `Destacam -se` vira `Destacam-se`. Na busca BM25, palavras como "qual",
"segundo" e "documento" são ignoradas para não favorecer trechos irrelevantes.
Se o modelo copia a frase certa mas indica o trecho vizinho (eles se sobrepõem),
a citação é corrigida para o trecho do mesmo PDF que a contém.

PDFs precisam ter texto extraível. PDFs escaneados precisam de OCR prévio;
imagens, gráficos e a disposição de tabelas não são interpretados visualmente.
Páginas sem texto e PDFs corrompidos ou com senha geram avisos. Perguntas que
dependem de muitos trechos ou de relações entre páginas podem precisar de ajustes.

### Medir a meta de 70–80%

Cada resposta exibe somente o texto completo e uma nota ao final, como `(0.75)`.
A nota é a **cobertura textual nas citações em escala de 0 a 1**, não uma
probabilidade de acerto. Fontes e indicadores continuam registrados internamente
e na saída de avaliação, sem aparecer no texto da resposta. Por padrão, a geração
usa 1 ou 2 parágrafos curtos, com linguagem simples e cerca de 60 a 120 palavras.
Perguntas simples podem receber apenas uma ou duas frases; explicações maiores
ficam para pedidos explícitos de detalhes. Condições essenciais são preservadas.
A precisão real permanece
`precision_percent: null` até existir uma avaliação de referência; o sistema
não transforma cosseno ou nota do modelo em probabilidade de acerto.

A cobertura é a proporção de termos distintos de cada afirmação presentes nas
suas citações literais, ignorando palavras gramaticais e diferenças de acento
e maiúsculas. Usa-se a menor cobertura entre as afirmações. Números e negações
são mantidos. Sinônimos podem reduzir essa cobertura mesmo em uma resposta
correta, e 100% de cobertura não garante que a conclusão esteja correta.
A similaridade registrada internamente é o menor dos melhores cossenos das fontes
citadas por afirmação, sem conversão em porcentagem. Respostas recusadas exibem
somente a recusa, sem nota inventada.

Não há garantia de precisão para PDFs ainda não avaliados. Similaridade dos
embeddings e temperatura zero **não são uma probabilidade de resposta correta**.
Crie um arquivo `perguntas.jsonl` com perguntas reais e respostas de referência,
incluindo perguntas cuja resposta não existe no documento:

```jsonl
{"question": "Qual é o prazo de reembolso?", "expected_answer": "Preencha com a resposta conferida no seu PDF e a página."}
{"question": "Qual é a senha de acesso?", "expected_answer": "Deve recusar se o PDF não informar a senha."}
```

```bash
./executar_rag.sh --avaliar perguntas.jsonl --saida .rag/avaliacao.jsonl
# Abra a saída, confira cada resposta e preencha "correct": true ou false.
./executar_rag.sh --relatorio .rag/avaliacao.jsonl
```

Se a avaliação parar no meio (por exemplo, por limite da API), repita o mesmo
comando: as perguntas já respondidas são mantidas e ela continua de onde parou.
Uma linha pode ter `"pdfs": ["a.pdf", "b.pdf"]` para avaliar perguntas sobre
dois documentos juntos.

Para medir só a **busca**, sem gastar cota da API, use as páginas esperadas:

```jsonl
{"question": "O que é desenvolvimento sustentável?", "pages": [4]}
{"question": "O que os dois contratos dizem sobre multa?", "pages": [2, 7], "pdfs": ["a.pdf", "b.pdf"]}
```

```bash
./executar_rag.sh --avaliar-busca busca.jsonl
```

`hit_rate` é a fração de perguntas em que alguma página esperada foi enviada ao
modelo; `first_place_rate`, em que ela veio em primeiro; `mrr` premia posições
altas. Com `"file"` na linha, a página só conta se for daquele PDF.

Marque `true` somente quando a resposta inteira estiver correta e sustentada
pelas páginas, ou quando uma recusa for realmente esperada. Marque `false` para
respostas incorretas, incompletas ou recusas a perguntas respondíveis. O relatório
mostra acerto entre os casos revisados, precisão entre respostas fornecidas e
cobertura; recusas não podem esconder falta de cobertura. A meta de 80% só é
calculada após revisar todas as linhas. Use pelo menos 30 perguntas variadas,
avalie PDFs representativos e reserve perguntas que não foram usadas em ajustes.

Implementação dos embeddings: [FastEmbed](https://qdrant.tech/documentation/embeddings/fastembed/),
com [Multilingual E5 small](https://huggingface.co/intfloat/multilingual-e5-small)
em [ONNX quantizado](https://huggingface.co/Xenova/multilingual-e5-small).

### O RAG como ferramenta do agente (`search_documents`)

O agente também usa o RAG, como **ferramenta**: `search_documents(query, files)` busca nos PDFs de `documentos/` com o
mesmo `PdfRag` e devolve os trechos mais relevantes, cada um com arquivo e página. Só a busca roda (local, sem API);
quem escreve a resposta é o próprio agente. A geração com citações validadas e a revisão acima são do chat do
`executar_rag.sh`, não da ferramenta. Assim cada busca vira um span `tool_call` no tracing, e o relatório mostra a
consulta e o tamanho do resultado (ex.: `search_documents: {"query": "prazo de reembolso"} → 5321 caracteres`).

```python
from workshop_agentes.tools.documents import make_search_documents

registry.register(make_search_documents())  # junto com update_plan, antes de criar o Agent
```

- **Saída:** `{"passages": [{"id", "file", "page", "text"}], "files": [...]}`, até 6 trechos por busca; sem trecho
  relevante, `passages` vazio e uma nota. Erros voltam ao modelo como observação: `NoDocuments` (nenhum PDF legível) e
  `UnknownFile` (um nome em `files` que não existe, com a lista dos disponíveis).
- Os trechos são **dados, nunca instruções**: a descrição da ferramenta diz isso ao modelo e pede que ele cite arquivo e
  página.
- No notebook, a seção **6** monta esse agente (indexa os PDFs antes do run e mostra os trechos recuperados), a **7**
  mostra o tracing e a memória do run e a **8** roda os evals. Na linha de comando, os evals registram a ferramenta sozinhos quando há dependências e PDFs (veja abaixo).

## Testes

Os testes são **offline**: usam um modelo falso, sem chave, internet ou cota.

```bash
.venv/bin/python -m pytest
```

## Avaliação (evals)

O módulo `workshop_agentes/evals` roda o agente num **golden set** (`workshop_agentes/evals/golden_set.json`) e confere
cada run: se **concluiu** (resposta final em texto), se houve **erro** de modelo ou de ferramenta, os **termos
esperados** na resposta (sem diferenciar maiúsculas e acentos), o número de **passos** e as **ferramentas** chamadas.
Opcionalmente, um **juiz LLM** compara a resposta com a rubrica do caso. As métricas vêm do tracing de cada run:
latência (total e sem as esperas por 429), tokens, chamadas ao modelo e retentativas (do agente e do juiz,
separadas e somadas). O resumo mostra a taxa de sucesso e de conclusão da suíte.

```bash
.venv/bin/python -m workshop_agentes.evals --dry-run      # valida o golden set e estima as chamadas; não usa a API
.venv/bin/python -m workshop_agentes.evals                # roda todos os casos (API real)
.venv/bin/python -m workshop_agentes.evals --only aritmetica busca-binaria
.venv/bin/python -m workshop_agentes.evals --judge        # + juiz nas rubricas (+1 chamada por caso com rubrica)
.venv/bin/python -m workshop_agentes.evals --rescore eval_results/<data-hora>.json   # reavalia sem rodar o agente
```

- **Saídas:** o relatório por caso e o resumo no terminal; em `eval_results/` (não versionada), o `.txt` com o tracing
  completo de cada run e o `.json` com os runs, que o `--rescore` reavalia depois de você mudar as expectativas, sem
  chamar a API (o juiz só roda de novo com `--judge`). O código de saída é 0 quando nenhum caso falha.
- **Custo:** cada caso é um run (≥ 2 chamadas: o plano e a resposta); os 5 casos gerais gastam ≈ 10 chamadas e, com o
  limite de saída por minuto (OTPM), levam alguns minutos. O `documentos-fonte` gasta ≥ 3 (plano, busca e resposta). Erro que se repetiria em todos os casos (chave inválida, modelo
  inexistente, cota diária) interrompe a suíte. Outras opções: `--model`, `--judge-model`, `--golden`, `--out` e
  `--quiet` (esconde as notas `→` do agente e os avisos de retentativa).
- **Formato de um caso:** `id`, `task` e, opcionais, `must_include` (cada item é um termo ou uma lista de alternativas),
  `must_not_include`, `max_steps`, `expected_tools`, `forbidden_tools` e `rubric` (em inglês, para o juiz). Campo
  desconhecido é erro, para um erro de digitação não desligar um check em silêncio.
- **Ferramentas novas:** registre-as em `build_registry()` (`workshop_agentes/evals/__main__.py`), como no notebook. Um
  caso com `"expected_tools": ["nome"]` fica **pulado** (sem chamar a API) enquanto a ferramenta não estiver
  registrada. Os checks de ferramenta leem os spans `tool_call`, então valem para qualquer ferramenta.
- **Documentos:** `search_documents` só é registrada quando as dependências do RAG estão instaladas e há algum PDF em
  `documentos/`. Sem isso, o caso `documentos-fonte` fica pulado (o `--dry-run` mostra). Ele vale para qualquer PDF:
  exige a busca e o nome do arquivo (`.pdf`) na resposta. O `aritmetica` tem `forbidden_tools: ["search_documents"]`:
  o agente não deve buscar à toa. `ppc-carga-horaria` usa o PPC de exemplo (3472 h, com a busca) e
  `acao-sem-ferramenta` pede para enviar um e-mail, que o agente deve recusar por não ter ferramenta para isso. Casos
  com o conteúdo dos seus PDFs podem ficar fora do git, em `documentos/casos.json` (coberto pelo `.gitignore`),
  rodados com `--golden documentos/casos.json`.
- No notebook, a seção **8** roda `ppc-carga-horaria`, `aritmetica` e `acao-sem-ferramenta` com o agente que tem
  `search_documents` e mostra o resultado em tabela.

## Notas

- **Modelo padrão:** `qwen/qwen3.8-27b` (`DEFAULT_MODEL` em `workshop_agentes/agent/config.py`), testado contra a API
  real na tarefa do notebook. É um modelo *Preview* do Groq (pode ser descontinuado com pouco aviso); nesse caso, defina
  `GROQ_MODEL` com outro modelo que aceite ferramentas. O secret ou variável `GROQ_MODEL` tem prioridade. Atenção:
  `openai/gpt-oss-120b` falhou nessa tarefa (erro 400 `tool_use_failed` ao chamar `finish` com uma resposta longa em
  Markdown). Se aparecer "modelo não encontrado", o notebook lista os modelos ativos para a sua chave.
- **Temperatura:** por padrão não é passada, e vale o padrão do `ChatGroq` (0.7).
- **Como o agente termina:** quando o modelo responde em texto, sem pedir ferramenta (`no_tool_call`), como no padrão da
  documentação do Groq; `length` indica que essa resposta foi cortada pelo limite de tokens. Antes a resposta ia dentro
  da ferramenta `finish`, e respostas longas em Markdown dentro do JSON causaram o `tool_use_failed` do gpt-oss e os
  cortes abaixo; `finish` continua disponível como ferramenta terminal opcional.
- **Primeiro passo forçado:** a 1ª chamada ao modelo usa `tool_choice` para obrigá-lo a chamar `update_plan`
  (`AgentConfig.first_tool`; `None` deixa o modelo decidir). Antes, com o modelo livre, ele às vezes pulava o plano
  (ex.: "Quanto é 2+2?" em 1 passo). Sem outras ferramentas, o run típico tem 2 passos (plano e resposta em texto);
  isso não é um limite (`max_steps=8`): com ferramentas que buscam ou executam algo, os passos aumentam.
- **Tamanho da resposta:** `build_llm` sempre envia `max_tokens=8192`. Sem limite ou com 2048, o Groq cortou a resposta
  dentro do `finish` em ~1000 caracteres, sem avisar (o `finish_reason` continuou `tool_calls`); com 8192 ela veio
  completa.
- **Limite de saída (OTPM):** além dos limites por requisição e por token, o qwen tem um limite de tokens de **saída** por
  minuto (1000 no plano gratuito testado). Ele não aparece nos cabeçalhos da resposta, só no erro 429; na prática cabe
  mais ou menos um run por minuto. Antes de gerar, o Groq reserva uma **estimativa** da saída (ex.: 816 tokens para uma
  chamada que gerou ~140), e a cota se recompõe aos poucos ao longo do minuto. O relatório de tracing mostra cada
  tentativa retentada com os números do limite (ex.: `RateLimitError OTPM: Limit 1000, Used 253, Requested 816`), e com
  `verbose=True` o agente avisa ao vivo quando está esperando.
- **Limites do plano gratuito:** cada passo do agente é uma chamada à API (`max_steps=8` por padrão) e reenvia o histórico
  inteiro, então os tokens de entrada crescem a cada passo; o limite de **tokens por minuto** é o que costuma estourar
  primeiro.
- **Retry com limite:** um 429 por minuto (RPM, TPM, OTPM) é retentado enquanto a espera somada couber em **60 s** por
  chamada (`RATE_LIMIT_BUDGET`, o tempo de a cota se recompor). Cada espera é a que a API pede (`retry-after`) ou o
  backoff, o que for maior. Se a próxima espera não couber, a chamada falha na hora em vez de travar o notebook. 5xx,
  falha de conexão e chamada de ferramenta malformada têm no máximo **3 tentativas** e **30 s** de espera cada (pior
  caso ≈ 1 min), mais o timeout de 30 s por requisição. Limite diário (RPD/TPD) esgotado falha na hora com
  `DailyQuotaError` (a detecção ainda será confirmada contra a API real).
