"""Visualizações do notebook do workshop: conexão, ferramentas, resultado, índice, trechos, tracing,
memória e evals, em HTML simples (sem dependências novas).

Cada show_* exibe no notebook (o IPython só é importado ao chamar); as funções *_html devolvem a
string, para os testes. Todo texto vem de fora (modelo, PDFs, ferramentas) e passa por html.escape.
O CSS herda a cor do texto e usa fundos translúcidos: funciona no tema claro e no escuro (Colab, VS Code).
"""

import html
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from workshop_agentes.agent.loop import AgentResult
from workshop_agentes.evals.checks import run_metrics
from workshop_agentes.evals.report import CHECK_MARKS, format_case, format_summary, summarize
from workshop_agentes.evals.runner import STATUS_LABELS, CaseResult
from workshop_agentes.llm import error_label, explain_error, invoke_with_retry, text_of
from workshop_agentes.tracing.report import PLAN_FIELDS, TOKEN_KEYS, format_report, preview_args, usage_of

CONNECTION_PROMPT = "Responda apenas: conexão funcionando!"
PASSAGE_PREVIEW = 280  # caracteres de cada trecho nos cartões
RESULT_PREVIEW = 1200  # caracteres de um resultado de ferramenta na memória (o resto fica recolhido)
ANSWER_PREVIEW = 600  # caracteres da resposta na tabela dos evals
REASON_KIND = {"no_tool_call": "ok", "finish": "ok", "length": "warn", "max_steps": "warn", "error": "bad"}
STATUS_KIND = {"pass": "ok", "fail": "bad", "skip": "warn", "not_run": "warn"}

STYLE = """<style>
.wa{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:13px;line-height:1.45;color:inherit;max-width:980px}
.wa .card{border:1px solid rgba(127,127,127,.35);border-radius:8px;padding:10px 12px;margin:6px 0;background:rgba(127,127,127,.06)}
.wa .chips{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 8px}
.wa .chip{border-radius:999px;padding:1px 9px;font-size:12px;background:rgba(127,127,127,.16);white-space:nowrap}
.wa .ok{background:rgba(34,160,90,.25)}.wa .bad{background:rgba(220,60,60,.25)}
.wa .warn{background:rgba(230,160,30,.28)}.wa .info{background:rgba(60,120,230,.22)}
.wa table{border-collapse:collapse;width:100%}
.wa th,.wa td{border-bottom:1px solid rgba(127,127,127,.25);padding:5px 8px;text-align:left;vertical-align:top}
.wa th{font-weight:600;opacity:.85}
.wa .bar{height:7px;border-radius:4px;background:rgba(60,120,230,.6);margin-top:3px}
.wa .bar.time{background:rgba(230,160,30,.7)}
.wa .muted{opacity:.72;font-size:12px}
.wa pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace}
.wa code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;background:rgba(127,127,127,.15);padding:0 4px;border-radius:4px}
.wa summary{cursor:pointer}
.wa .msg{border-left:4px solid rgba(127,127,127,.5);padding:6px 10px;margin:6px 0;background:rgba(127,127,127,.06);border-radius:0 6px 6px 0}
.wa .msg.user{border-color:rgba(60,120,230,.85)}.wa .msg.ai{border-color:rgba(150,90,220,.85)}
.wa .msg.tool{border-color:rgba(230,160,30,.9)}.wa .msg.final{border-color:rgba(34,160,90,.95)}
.wa .who{font-weight:600}
.wa .passage{border:1px solid rgba(127,127,127,.3);border-radius:6px;padding:6px 9px;margin:5px 0}
</style>"""


# --- Exibição ------------------------------------------------------------------------


def _show(html_text: str) -> None:
    from IPython.display import HTML, display  # só no notebook

    display(HTML(html_text))


def _show_markdown(text: str) -> None:
    from IPython.display import Markdown, display

    display(Markdown(text))


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _wrap(body: str) -> str:
    return f"<div class='wa'>{STYLE}{body}</div>"


def _chip(label: Any, kind: str = "") -> str:
    return f"<span class='chip {kind}'>{_esc(label)}</span>"


def _chips(*chips: str) -> str:
    return f"<div class='chips'>{''.join(chips)}</div>"


def _bar(value: float, maximum: float, kind: str = "") -> str:
    width = 0 if maximum <= 0 else max(2.0, min(100.0, 100 * value / maximum))
    return f"<div class='bar {kind}' style='width:{width:.0f}%'></div>"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _parse(content: Any) -> Any:
    """Conteúdo de um ToolMessage: o dict, se for JSON; senão o texto."""
    text = text_of(content) if not isinstance(content, str) else content
    try:
        return json.loads(text)
    except ValueError:
        return text


# --- 2. Conexão ----------------------------------------------------------------------


def check_connection(llm: Any, api_key: str | None = None) -> bool:
    """Uma pergunta, uma resposta (o "chatbot"): mostra a resposta ou o que fazer se falhar.

    Passa pelo invoke_with_retry, o mesmo retry com limite do agente. Na falha, mostra só o rótulo
    seguro do erro (error_label) e a explicação (explain_error), nunca a mensagem.
    """
    try:
        reply, attempts = invoke_with_retry(llm, [HumanMessage(content=CONNECTION_PROMPT)])
    except Exception as exc:
        _show(connection_error_html(exc, api_key))
        return False
    _show(connection_html(reply, attempts))
    return True


def connection_html(reply: AIMessage, attempts: int) -> str:
    usage = reply.usage_metadata or {}
    chips = _chips(
        _chip("conectado", "ok"),
        _chip(f"tokens: entrada {usage.get('input_tokens', 0)} · saída {usage.get('output_tokens', 0)}"),
        _chip(f"tentativas: {attempts}"),
        _chip(f"finish_reason: {reply.response_metadata.get('finish_reason')}"),
    )
    body = f"<div class='muted'>Pergunta: {_esc(CONNECTION_PROMPT)}</div><div>{_esc(text_of(reply.content))}</div>"
    return _wrap(f"<div class='card'>{chips}{body}</div>")


def connection_error_html(exc: BaseException, api_key: str | None = None) -> str:
    chips = _chips(_chip("falhou", "bad"), _chip(error_label(exc)))
    return _wrap(f"<div class='card'>{chips}<div>{_esc(explain_error(exc, api_key))}</div></div>")


# --- 3. Ferramentas ------------------------------------------------------------------


def show_tools(agent: Any) -> None:
    """Tabela das ferramentas que o modelo pode pedir (agent ou ToolRegistry)."""
    _show(tools_html(agent))


def tools_html(agent: Any) -> str:
    registry = getattr(agent, "registry", agent)
    first_tool = getattr(getattr(agent, "cfg", None), "first_tool", None)
    rows = []
    for tool in registry.tools:
        forced = _chip("1ª chamada obrigatória", "info") if tool.name == first_tool else ""
        args = ", ".join(f"<code>{_esc(name)}</code>" for name in tool.args) or "-"
        rows.append(
            f"<tr><td><code>{_esc(tool.name)}</code> {forced}</td>"
            f"<td>{_esc(tool.description)}</td><td>{args}</td></tr>"
        )
    count = len(registry.tools)
    header = f"<div class='who'>{count} ferramenta{'s' if count != 1 else ''} que o modelo pode pedir</div>"
    table = (
        "<table><tr><th>Ferramenta</th><th>O que faz (a descrição que o modelo lê)</th><th>Argumentos</th></tr>"
        f"{''.join(rows)}</table>"
    )
    return _wrap(f"<div class='card'>{header}{table}</div>")


# --- 4 a 6. Resultado de um run ------------------------------------------------------


def show_result(result: AgentResult) -> None:
    """Motivo de parada, passos, tokens e duração; depois a resposta (Markdown, já escapado)."""
    _show(result_html(result))
    if result.answer:
        _show_markdown(answer_markdown(result))


def result_html(result: AgentResult) -> str:
    m = run_metrics(result)
    chips = [
        _chip(f"motivo: {result.reason}", REASON_KIND.get(result.reason, "")),
        _chip(f"passos: {result.steps}"),
        _chip(f"chamadas ao modelo: {m['llm_calls']}"),
        _chip(f"ferramentas: {', '.join(m['tools_used']) or '-'}"),
        _chip(f"tokens: entrada {m['input_tokens']} · saída {m['output_tokens']}"),
    ]
    if m["run_latency"] is not None:
        chips.append(_chip(f"duração: {m['run_latency']:.1f}s"))
    if m["rate_limit_wait"]:
        chips.append(_chip(f"esperando limite (429): {m['rate_limit_wait']:.1f}s", "warn"))
    if result.error:
        chips.append(_chip(f"erro: {result.error}", "bad"))
    note = "" if result.answer else "<div class='muted'>Sem resposta final neste run.</div>"
    return _wrap(f"<div class='card'>{_chips(*chips)}{note}</div>")


def answer_markdown(result: AgentResult) -> str:
    """A resposta como Markdown, com o HTML escapado: a formatação aparece e nenhuma tag é interpretada."""
    return html.escape(result.answer or "", quote=False)


# --- 6. RAG: índice e trechos --------------------------------------------------------


def show_index(engine: Any, seconds: float | None = None) -> None:
    """O que foi indexado: PDFs, páginas com texto, trechos e avisos."""
    _show(index_html(engine, seconds))


def index_html(engine: Any, seconds: float | None = None) -> str:
    pages = len({(chunk.file, chunk.page) for chunk in engine.chunks})
    chips = [
        _chip(f"PDFs: {len(engine.files)}", "info"),
        _chip(f"páginas com texto: {pages}"),
        _chip(f"trechos: {len(engine.chunks)}"),
    ]
    if seconds is not None:
        chips.append(_chip(f"tempo: {seconds:.1f}s"))
    files = "".join(f"<div><code>{_esc(name)}</code></div>" for name in engine.files) or "<div>Nenhum PDF.</div>"
    warnings = "".join(f"<div class='muted'>⚠ {_esc(w)}</div>" for w in engine.warnings)
    return _wrap(f"<div class='card'>{_chips(*chips)}{files}{warnings}</div>")


def show_passages(result: AgentResult) -> None:
    """Os trechos que cada busca do run devolveu ao modelo, com arquivo e página."""
    _show(passages_html(result))


def passages_html(result: AgentResult) -> str:
    searches = _searches(result.messages)
    if not searches:
        return _wrap("<div class='card muted'>Nenhuma busca neste run.</div>")
    blocks = []
    for number, (args, out) in enumerate(searches, start=1):
        query = args.get("query") if isinstance(args, dict) else None
        passages = out.get("passages") or []
        head = _chips(_chip(f"busca {number}", "info"), _chip(f"{len(passages)} trechos"))
        cards = "".join(_passage_html(p) for p in passages) or f"<div class='muted'>{_esc(out.get('note', ''))}</div>"
        blocks.append(f"<div class='card'>{head}<div>Consulta: <b>{_esc(query or '-')}</b></div>{cards}</div>")
    return _wrap("".join(blocks))


def _passage_html(passage: dict) -> str:
    text = str(passage.get("text", ""))
    preview = text if len(text) <= PASSAGE_PREVIEW else text[:PASSAGE_PREVIEW] + "…"
    where = f"{passage.get('file', '?')} · p. {passage.get('page', '?')}"
    return f"<div class='passage'><div class='muted'>{_esc(where)}</div>{_esc(preview)}</div>"


def _searches(messages: Sequence[BaseMessage]) -> list[tuple[Any, dict]]:
    """(args da chamada, resultado) de cada ferramenta que devolveu "passages", na ordem."""
    args_by_id = {
        call.get("id"): call.get("args") for m in messages if isinstance(m, AIMessage) for call in m.tool_calls
    }
    found = []
    for m in messages:
        if isinstance(m, ToolMessage):
            out = _parse(m.content)
            if isinstance(out, dict) and "passages" in out:
                found.append((args_by_id.get(m.tool_call_id, {}), out))
    return found


# --- 7. Bastidores: tracing ----------------------------------------------------------


def show_trace(result: AgentResult) -> None:
    """Linha do tempo do run: o que aconteceu em cada passo, contexto (tokens de entrada) e tempo."""
    _show(trace_html(result))


def trace_html(result: AgentResult) -> str:
    spans = result.spans
    children = defaultdict(list)
    for s in spans:
        children[s.parent_id].append(s)
    steps = [s for s in spans if s.kind == "step"]
    max_in = max((usage_of(s)["input_tokens"] for s in spans if s.kind == "llm_call"), default=0)
    max_time = max((s.latency or 0.0 for s in steps), default=0.0)
    rows = []
    for number, step in enumerate(steps, start=1):
        kids = children[step.id]
        llm_calls = [s for s in kids if s.kind == "llm_call"]
        tool_calls = [s for s in kids if s.kind == "tool_call"]
        usage = {key: sum(usage_of(s)[key] for s in llm_calls) for key in TOKEN_KEYS}
        events = [_tool_event(s) for s in tool_calls] or [_text_event(llm_calls, result, number == len(steps))]
        forced = [s.attrs["forced_tool"] for s in llm_calls if s.attrs.get("forced_tool")]
        if forced:
            events.insert(0, _chip(f"tool_choice forçado: {forced[0]}", "info"))
        retries = [note for s in llm_calls for note in s.attrs.get("retries") or []]
        latency = step.latency or 0.0
        timing = f"{latency:.2f}s{_bar(latency, max_time, 'time')}"
        if retries:
            timing += "".join(f"<div class='muted'>retentado: {_esc(note)}</div>" for note in retries)
        rows.append(
            f"<tr><td><b>{number}</b></td><td>{''.join(events)}</td>"
            f"<td>{usage['input_tokens']}{_bar(usage['input_tokens'], max_in)}</td>"
            f"<td>{usage['output_tokens']}</td><td>{timing}</td></tr>"
        )
    m = run_metrics(result)
    chips = [
        _chip(f"passos: {len(steps)}"),
        _chip(f"chamadas ao modelo: {m['llm_calls']}"),
        _chip(f"chamadas de ferramenta: {m['tool_calls']}"),
        _chip(f"tokens: entrada {m['input_tokens']} · saída {m['output_tokens']}"),
        _chip(f"motivo: {result.reason}", REASON_KIND.get(result.reason, "")),
    ]
    if m["run_latency"] is not None:
        chips.append(_chip(f"duração: {m['run_latency']:.1f}s (sem esperas de 429: {m['net_latency']:.1f}s)"))
    table = (
        "<table><tr><th>Passo</th><th>O que aconteceu</th><th>Entrada (contexto)</th><th>Saída</th><th>Tempo</th></tr>"
        f"{''.join(rows)}</table>"
    )
    caption = (
        "<div class='muted'>Entrada = todo o histórico reenviado ao modelo naquele passo: por isso cresce a cada "
        "passo (a API não guarda nada entre as chamadas). O tempo inclui as esperas de retentativa.</div>"
    )
    report = f"<details><summary class='muted'>Relatório completo (texto)</summary><pre>{_esc(format_report(spans))}</pre></details>"
    return _wrap(f"<div class='card'>{_chips(*chips)}{table}{caption}{report}</div>")


def _tool_event(span: Any) -> str:
    """Uma chamada de ferramenta: o nome e o plano (update_plan) ou os args e o tamanho do resultado."""
    args = span.attrs.get("args")
    head = _chip(f"ferramenta: {span.name}", "bad" if "error" in span.attrs else "info")
    if isinstance(args, dict) and any(key in args for key in PLAN_FIELDS):
        detail = f"<div class='muted'>plano: {_esc(args.get('plan', '-'))}</div>"
        detail += f"<div class='muted'>próximo passo: {_esc(args.get('next_step', '-'))}</div>"
    else:
        size = span.attrs.get("result_chars")
        detail = f"<div><code>{_esc(preview_args(args))}</code>"
        detail += f" <span class='muted'>→ {size} caracteres</span></div>" if size is not None else "</div>"
    if "error" in span.attrs:
        detail += f"<div class='muted'>rejeitada: {_esc(span.attrs['error'])}</div>"
    return f"<div>{head}{detail}</div>"


def _text_event(llm_calls: Sequence[Any], result: AgentResult, last: bool) -> str:
    """Um passo sem ferramenta: a resposta final, uma resposta vazia ou um erro na chamada."""
    errors = [s.attrs["error"] for s in llm_calls if "error" in s.attrs]
    if errors:
        return _chip(f"erro na chamada: {errors[0]}", "bad")
    if any(s.attrs.get("empty_reply") for s in llm_calls):
        return _chip("resposta vazia: o loop pediu de novo", "warn")
    if last and result.answer:
        return _chip("resposta final em texto", "ok")
    return _chip("texto")


# --- 7. Bastidores: memória ----------------------------------------------------------


def show_memory(result: AgentResult) -> None:
    """A memória do run como conversa, agrupada por passo: o que o modelo recebeu a cada chamada."""
    _show(memory_html(result))


def memory_html(result: AgentResult) -> str:
    messages = result.messages
    calls = [s for s in result.spans if s.kind == "llm_call" and "error" not in s.attrs]  # um por AIMessage
    names = {call.get("id"): call.get("name") for m in messages if isinstance(m, AIMessage) for call in m.tool_calls}
    parts, step, users = [], 0, 0
    for m in messages:
        if isinstance(m, SystemMessage):
            text = text_of(m.content)
            parts.append(
                f"<div class='msg'><details><summary><span class='who'>System prompt</span> "
                f"<span class='muted'>instruções fixas, reenviadas em todo passo ({len(text)} caracteres)</span>"
                f"</summary><pre>{_esc(text)}</pre></details></div>"
            )
        elif isinstance(m, HumanMessage):
            users += 1
            who = "User · a tarefa" if users == 1 else "Observação do loop"
            parts.append(f"<div class='msg user'><div class='who'>{who}</div>{_esc(text_of(m.content))}</div>")
        elif isinstance(m, AIMessage):
            usage = usage_of(calls[step]) if step < len(calls) else None
            step += 1
            parts.append(_ai_html(m, step, usage))
        elif isinstance(m, ToolMessage):
            parts.append(_tool_result_html(m, names.get(m.tool_call_id) or m.name or "?"))
    footer = (
        f"<div class='muted'>{len(messages)} mensagens. A memória vale só durante este run: o próximo run() "
        "começa de novo, só com o system prompt.</div>"
    )
    return _wrap("".join(parts) + footer)


def _ai_html(m: AIMessage, step: int, usage: dict | None) -> str:
    tokens = f" <span class='muted'>· recebeu {usage['input_tokens']} tokens, gerou {usage['output_tokens']}</span>" if usage else ""
    text = text_of(m.content).strip()
    calls = [*m.tool_calls, *m.invalid_tool_calls]
    if not calls:
        return f"<div class='msg final'><div class='who'>Passo {step} · modelo: resposta final{tokens}</div><pre>{_esc(text)}</pre></div>"
    body = f"<div class='muted'>texto junto da chamada (não é a resposta): {_esc(text)}</div>" if text else ""
    for call in calls:
        args = call.get("args")
        if isinstance(args, dict) and any(key in args for key in PLAN_FIELDS):
            fields = "".join(f"<div><span class='muted'>{_esc(k)}:</span> {_esc(v)}</div>" for k, v in args.items())
        else:
            fields = f"<pre>{_esc(_json(args))}</pre>"
        body += f"<div>pede <code>{_esc(call.get('name') or '?')}</code>{fields}</div>"
    return f"<div class='msg ai'><div class='who'>Passo {step} · modelo{tokens}</div>{body}</div>"


def _tool_result_html(m: ToolMessage, name: str) -> str:
    out = _parse(m.content)
    head = f"<div class='who'>Resultado de <code>{_esc(name)}</code> <span class='muted'>· volta ao modelo como observação</span></div>"
    if isinstance(out, dict) and isinstance(out.get("passages"), list):
        passages = out["passages"]
        where = ", ".join(f"p. {p.get('page')}" for p in passages[:8]) + ("…" if len(passages) > 8 else "")
        summary = f"{len(passages)} trechos ({where}) · {len(m.content)} caracteres"
        body = f"<details><summary>{_esc(summary)}</summary>{''.join(_passage_html(p) for p in passages)}</details>"
    elif isinstance(out, dict) and "error" in out:
        body = _chip(f"erro: {out['error']}", "bad")
    else:
        text = _json(out) if isinstance(out, (dict, list)) else str(out)
        if len(text) > RESULT_PREVIEW:
            body = (
                f"<pre>{_esc(text[:RESULT_PREVIEW])}…</pre>"
                f"<details><summary class='muted'>ver o resultado inteiro</summary><pre>{_esc(text)}</pre></details>"
            )
        else:
            body = f"<pre>{_esc(text)}</pre>"
    return f"<div class='msg tool'>{head}{body}</div>"


# --- 8. Evals ------------------------------------------------------------------------


def show_evals(results: Sequence[CaseResult]) -> None:
    """Tabela dos casos (status, checks, passos, latência, tokens) e o resumo da suíte."""
    _show(evals_html(results))


def evals_html(results: Sequence[CaseResult]) -> str:
    s = summarize(results)
    chips = [
        _chip(f"passaram: {s['passed']}/{s['executed']}", "ok" if s["executed"] and not s["failed"] else "bad"),
        _chip(f"pulados: {s['skipped']}", "warn" if s["skipped"] else ""),
        _chip(f"chamadas ao modelo: {s['llm_calls']}"),
        _chip(f"tokens: {s['total_tokens']}"),
        _chip(f"retentativas: {s['retries_total']}", "warn" if s["retries_total"] else ""),
    ]
    rows = "".join(_case_row(r) for r in results)
    table = (
        "<table><tr><th>Caso</th><th>Status</th><th>Checks</th><th>Run</th></tr>"
        f"{rows}</table>"
    )
    text = "\n\n".join([*(format_case(r) for r in results), format_summary(results)])
    report = f"<details><summary class='muted'>Relatório completo (texto)</summary><pre>{_esc(text)}</pre></details>"
    return _wrap(f"<div class='card'>{_chips(*chips)}{table}{report}</div>")


def _case_row(r: CaseResult) -> str:
    status = _chip(STATUS_LABELS[r.status], STATUS_KIND.get(r.status, ""))
    note = f"<div class='muted'>{_esc(r.note)}</div>" if r.note else ""
    checks = "".join(
        f"<div>{CHECK_MARKS[c.passed]} {_esc(c.name)}" + (f" <span class='muted'>{_esc(c.detail)}</span>" if c.detail else "") + "</div>"
        for c in r.checks
    )
    run = "-"
    if r.result is not None:
        m = run_metrics(r.result)
        latency = (
            f"<div>{m['run_latency']:.1f}s <span class='muted'>(sem 429: {m['net_latency']:.1f}s)</span></div>"
            if m["run_latency"] is not None else ""
        )
        answer = r.result.answer or "-"
        answer = answer if len(answer) <= ANSWER_PREVIEW else answer[:ANSWER_PREVIEW] + "…"
        run = (
            f"<div>{r.result.steps} passos · {m['total_tokens']} tokens</div>{latency}"
            f"<details><summary class='muted'>resposta</summary><pre>{_esc(answer)}</pre></details>"
        )
    task = f"<div class='muted'>{_esc(r.case.task)}</div>"
    return f"<tr><td><code>{_esc(r.case.id)}</code>{task}</td><td>{status}{note}</td><td>{checks}</td><td>{run}</td></tr>"
