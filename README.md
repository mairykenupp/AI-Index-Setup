# AI-Index-Setup

## Para usar a função de processamento simultâneo
Por padrão o servidor Ollama processa uma requisição de embedding por vez para um mesmo modelo, então threads concorrentes vão só enfileirar no servidor sem ganho real de GPU. Pra ele de fato aceitar chamadas em paralelo, defina antes de executar o Ollama:
Um de cada vez:

#### Windows:

No PowerShell:
```setx OLLAMA_NUM_PARALLEL 8```

```setx OLLAMA_MAX_LOADED_MODELS 1```

Reinicie o serviço Ollama:

```powershell
taskkill /IM "ollama.exe" /F
taskkill /IM "ollama app.exe" /F
```

(o segundo é o ícone da bandeja — se não existir esse processo, o comando só vai mostrar um erro inofensivo dizendo que não achou, pode ignorar)

#### Linux:

Se o Ollama roda como serviço do systemd (mais comum em instalação padrão Linux):

```sudo systemctl edit ollama.service```

Isso abre um editor. Adicione:

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=8"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
```

Salve e feche, depois:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Pra conferir se pegou:

```bash
systemctl show ollama --property=Environment
```

Se você roda o Ollama manualmente pelo terminal (ollama serve direto, sem ser como serviço), aí é só exportar antes de rodar:

```bash
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve
```

Isso vale só pra aquela sessão de terminal. Pra deixar permanente nesse caso, adicione as duas linhas export ... no final do seu ~/.bashrc (ou ~/.zshrc, se usar zsh).

Se não tiver certeza de qual dos dois é o seu caso, roda:

```bash
systemctl status ollama
```

Se aparecer algo tipo "active (running)", é o primeiro cenário (serviço).

#### Barras de progresso

Se quiser mais ou menos barras na tela ao mesmo tempo, é só mudar esse número na hora de rodar, dentro do arquivo ingest.py ou ingest_simultaneo.py:

`python ingest.py --file-workers 4`






