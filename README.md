# workshop-agentes

Material do workshop **"Do Chatbot ao Agente: Construindo Sistemas de IA com Ferramentas"**.

Um projeto mínimo e modular que mostra a evolução de uma aplicação com LLM para um **agente** que usa ferramentas e
raciocina em passos, mantendo o contexto até entregar o resultado:

- **modelo** (Groq via `langchain-groq`), que só *pede* as chamadas de ferramenta; quem executa é o nosso código;
- **memória de trabalho**: a API é *stateless*, então o histórico completo é reenviado a cada passo;
- **registry de ferramentas**: `update_plan` (raciocínio explícito) e `finish` (ferramenta terminal);
- **loop manual** de ferramentas, sem agentes prontos, para enxergar cada passo;
- **tracing** próprio: tokens, latência, `finish_reason` e plano por passo.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedroop32/workshop-agentes/blob/main/workshop.ipynb)

## Estrutura

```
workshop_agentes/
  llm.py              build_llm, invoke_with_retry (único ponto de retry), text_of, list_models
  agent/config.py     DEFAULT_MODEL, get_secret, AgentConfig
  agent/memory.py     WorkingMemory (histórico + invariante chamada -> resultado)
  agent/loop.py       Agent e AgentResult (o loop)
  tools/registry.py   ToolRegistry (executa ferramentas sem nunca levantar exceção)
  tools/core.py       update_plan e finish
  tracing/tracer.py   Span e Tracer
  tracing/report.py   relatório em texto
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

Requer Python 3.10+.

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
from workshop_agentes.tools.core import finish, update_plan
from workshop_agentes.tools.registry import ToolRegistry
from workshop_agentes.tracing.report import format_report
from workshop_agentes.tracing.tracer import Tracer

config = AgentConfig()
llm = build_llm(get_secret("GROQ_MODEL", DEFAULT_MODEL), get_secret("GROQ_API_KEY"), config.temperature)
registry = ToolRegistry()
registry.register(update_plan)
registry.register(finish, terminal=True)
agent = Agent(llm, registry, WorkingMemory(config.system_prompt), Tracer(), config)

result = agent.run("Compare lista encadeada e array dinâmico para implementar uma fila e conclua qual usar.")
print(result.reason, result.answer)
print(format_report(result.spans))
```

## Testes

Os testes são **offline**: usam um modelo falso, sem chave, internet ou cota.

```bash
.venv/bin/python -m pytest
```

## Notas

- **Modelo padrão:** `qwen/qwen3.8-27b` (`DEFAULT_MODEL` em `workshop_agentes/agent/config.py`), testado contra a API
  real na tarefa do notebook. É um modelo *Preview* do Groq (pode ser descontinuado com pouco aviso); nesse caso, defina
  `GROQ_MODEL` com outro modelo que aceite ferramentas. O secret ou variável `GROQ_MODEL` tem prioridade. Atenção:
  `openai/gpt-oss-120b` falhou nessa tarefa (erro 400 `tool_use_failed` ao chamar `finish` com uma resposta longa em
  Markdown). Se aparecer "modelo não encontrado", o notebook lista os modelos ativos para a sua chave.
- **Temperatura:** por padrão não é passada, e vale o padrão do `ChatGroq` (0.7).
- **Limites do plano gratuito:** cada passo do agente é uma chamada à API (`max_steps=8` por padrão) e reenvia o histórico
  inteiro, então os tokens de entrada crescem a cada passo; o limite de **tokens por minuto** é o que costuma estourar
  primeiro.
- **Retry com limite:** 429 por minuto, 5xx, falha de conexão e chamada de ferramenta malformada são retentados com
  backoff, no máximo **3 tentativas** por chamada e **30 s** de espera cada (pior caso ≈ 1 min por chamada, mais o
  timeout de 30 s por requisição). Se a API pedir uma espera maior que 30 s, a chamada falha na hora em vez de travar o
  notebook. Limite diário (RPD/TPD) esgotado falha na hora com `DailyQuotaError` (a detecção ainda será confirmada contra
  a API real).
