# workshop-agentes

Material do workshop **"Do Chatbot ao Agente: Construindo Sistemas de IA com Ferramentas"**.

Um projeto mínimo e modular que mostra a evolução de uma aplicação com LLM para um **agente** que usa ferramentas e
raciocina em passos, mantendo o contexto até entregar o resultado:

- **modelo** (Groq via `langchain-groq`), que só *pede* as chamadas de ferramenta; quem executa é o nosso código;
- **memória de trabalho**: a API é *stateless*, então o histórico completo é reenviado a cada passo;
- **registry de ferramentas**: `update_plan` (raciocínio explícito; a 1ª chamada é obrigada a usá-la); a resposta final
  vem em texto (`finish`, uma ferramenta terminal, é opcional);
- **loop manual** de ferramentas, sem agentes prontos, para enxergar cada passo;
- **tracing** próprio: tokens, latência, `finish_reason`, tentativas retentadas, plano por passo e motivo de parada.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedroop32/workshop-agentes/blob/main/workshop.ipynb)

## Estrutura

```
workshop_agentes/
  llm.py              build_llm, invoke_with_retry (único ponto de retry), error_label, text_of, list_models
  agent/config.py     DEFAULT_MODEL, get_secret, AgentConfig
  agent/memory.py     WorkingMemory (histórico + invariante chamada -> resultado)
  agent/loop.py       Agent e AgentResult (o loop)
  tools/registry.py   ToolRegistry (executa ferramentas sem nunca levantar exceção)
  tools/core.py       update_plan e finish (terminal opcional)
  tracing/tracer.py   Span e Tracer
  tracing/report.py   relatório em texto
  evals/              golden set, checks, juiz LLM opcional, runner, relatório e CLI (python -m workshop_agentes.evals)
tests/                testes offline (pytest) com modelo falso
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
4. Rode as células 1 a 5 em ordem.

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
latência, tokens, chamadas ao modelo e retentativas. O resumo mostra a taxa de sucesso e de conclusão da suíte.

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
- **Custo:** cada caso é um run (≥ 2 chamadas: o plano e a resposta); os 5 casos gastam ≈ 10 chamadas e, com o limite
  de saída por minuto (OTPM), levam alguns minutos. Erro que se repetiria em todos os casos (chave inválida, modelo
  inexistente, cota diária) interrompe a suíte. Outras opções: `--model`, `--judge-model`, `--golden`, `--out` e
  `--quiet` (esconde as notas `→` do agente e os avisos de retentativa).
- **Formato de um caso:** `id`, `task` e, opcionais, `must_include` (cada item é um termo ou uma lista de alternativas),
  `must_not_include`, `max_steps`, `expected_tools`, `forbidden_tools` e `rubric` (em inglês, para o juiz). Campo
  desconhecido é erro, para um erro de digitação não desligar um check em silêncio.
- **Ferramentas novas:** registre-as em `build_registry()` (`workshop_agentes/evals/__main__.py`), como no notebook. Um
  caso com `"expected_tools": ["nome"]` fica **pulado** (sem chamar a API) enquanto a ferramenta não estiver
  registrada. Os checks de ferramenta leem os spans `tool_call`, então valem para qualquer ferramenta.
- No notebook, o extra **E5** roda a mesma avaliação com o `agent` da seção 3 (por padrão, só o caso `aritmetica`).

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
