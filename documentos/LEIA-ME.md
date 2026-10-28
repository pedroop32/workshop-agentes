# Coloque seus PDFs aqui

Copie um ou mais arquivos `.pdf` para esta pasta (subpastas também são lidas).
Depois execute `./executar_rag.sh` na raiz do projeto.

O sistema atualiza o índice antes de cada pergunta. Dentro do chat, `/listar`
mostra os documentos e `/sair` encerra. Use `--pdf nome.pdf` para consultar
somente um documento, ou repita para ler vários juntos:
`--pdf a.pdf --pdf b.pdf`. No chat, use `/pdf a.pdf b.pdf`. PDFs novos, alterados e removidos são detectados.

No notebook, o extra E6 usa estes mesmos PDFs na ferramenta `search_documents` do agente.

Os PDFs precisam ter texto selecionável. Digitalizações sem camada de texto
precisam passar por OCR antes; páginas vazias e ilegíveis geram avisos.

Seus PDFs não são versionados. Os trechos recuperados são enviados à API Groq
para gerar a resposta. O índice e o modelo de busca ficam em `.rag/` localmente.
