# SofaScore Scraper — Brasileirão Série A 2025

Crawler modular e testável que extrai **estatísticas individuais partida a partida**
de todos os jogadores do Campeonato Brasileiro Série A 2025 a partir da API
interna do SofaScore (`api.sofascore.com/api/v1`), exportando para CSV em `data/`.

> **Aviso ético/acadêmico.** A API do SofaScore é privada e seu uso automatizado
> provavelmente viola os Termos de Serviço do site. Este código existe para fins
> acadêmicos (TCC). Use com volume baixo, delay gentil e nunca redistribua os
> dados. Respeite `robots.txt` e interrompa o uso se receber bloqueios (403/429).

## Stack

- `httpx` — requisições HTTP com timeouts e headers de navegador.
- `tenacity` — retry exponencial com backoff/jitter para 429 e erros 5xx.
- `pydantic` v2 — validação estrita de tipos de cada registro extraído.
- `pandas` — pré-processamento do dataset para modelagem.
- `pytest` — testes offline sobre payloads reais congelados em `tests/fixtures/`.

## Instalação

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Uso

```bash
# Temporada 2025 completa (rodadas 1-38), particionado por rodada + CSV consolidado
./.venv/bin/python scrape.py

# Rodadas específicas, sem partition, sem resume, delays customizados
./.venv/bin/python scrape.py --from-round 1 --to-round 1 --no-partition --no-resume \
    --delay-min 1.2 --delay-max 2.4
```

Saída (criada automaticamente):

```
data/
├── brasileirao_2025_jogadores_partida_a_partida.csv   # consolidado (utf-8-sig)
├── rodada_01.csv ... rodada_38.csv                     # particionado (opcional)
└── checkpoint.json                                     # resume por rodada
```

### Reproducibilidade

- **Resume:** após cada rodada o checkpoint é salvo; uma execução interrompida
  continua da rodada seguinte (`--no-resume` recomeça do zero).
- **Determinismo:** linhas ordenadas por `(match_date, match_id, team_id, player_id)`
  e header fixo definido pela ordem dos campos do modelo Pydantic — duas execuções
  idênticas geram o mesmo arquivo.
- **Fixtures reais:** `tests/fixtures/` guarda payloads autênticos da rodada 1/2025,
  então a suíte roda 100% offline, sem rede.

## Pré-processamento

Gera o dataset para modelagem a partir do CSV consolidado, sem nenhuma requisição à API:

```bash
./.venv/bin/python clean.py   # → data/processed/brasileirao_2025_dataset_modelo.csv
```

Etapas, regras de valores ausentes, dicionário das colunas novas e números medidos:
[`docs/pre_processamento.md`](docs/pre_processamento.md). Organização do código e onde
estender: [`docs/arquitetura.md`](docs/arquitetura.md).

## Modelagem (proposta)

Comparação entre um modelo genérico e um modelo por posição (G, D, M, F) para prever a nota:
desenho experimental, validação e arquitetura em [`docs/modelagem.md`](docs/modelagem.md).

## Testes

```bash
./.venv/bin/python -m pytest -q
```

## Como funciona

1. Resolve o `season_id` do ano desejado (`/unique-tournament/325/seasons`).
2. Para cada rodada: `/unique-tournament/325/season/{sid}/events/round/{n}` e filtra
   partidas com `status.type == "finished"`.
3. Para cada partida: `/event/{id}/lineups` (estatísticas por jogador) e
   `/event/{id}/incidents` (cartões amarelos/vermelhos por jogador e minuto).
4. Normaliza em registros tipados (`PlayerMatchStat`, Pydantic) e escreve o CSV.
   Substitutos não utilizados (sem `minutesPlayed`) são descartados — a linha
   contém apenas **titulares e reservas que entraram em campo**.

## Estrutura

Uma pasta por etapa. Detalhes em [`docs/arquitetura.md`](docs/arquitetura.md).

```text
sofascore_scraper/
├── scrape.py              # 1. coleta            → data/*.csv
├── clean.py               # 2. pré-processamento → data/processed/
├── train_history.py       # 3. modelagem (pré-jogo, o do TCC) → data/results_history*/
├── src/
│   ├── config.py          # constantes: URLs, tournament/season, delays, caminhos
│   ├── scraping/          # client, headers, rate_limiter, models, parsers, crawler, csv_writer
│   ├── cleaning/          # cleaners (etapas), pipeline (lê o bruto, grava o limpo)
│   └── modeling/          # features, history, estimators, evaluation, training, training_history
├── tests/                 # testes automatizados, uma pasta por etapa
│   ├── scraping/          # conftest.py e fixtures/ (payloads reais da rodada 1/2025)
│   ├── cleaning/
│   └── modeling/
├── docs/                  # arquitetura.md, pre_processamento.md, modelagem.md
├── data/
└── requirements.txt
```

## Estendendo para outros anos

Série A é sempre o `unique-tournament` 325 (constante em `src/config.py`); o ano é
descoberto dinamicamente. Para 2024:

```bash
./.venv/bin/python scrape.py --year 2024
```

Para um torneio diferente (ex.: Copa do Brasil, Série B), troque `TOURNAMENT_ID` no
`config.py` ou informe `--tournament <id>`.

## Dicionário das colunas (destaques)

| Grupo | Colunas |
|---|---|
| Partida | `match_id`, `round`, `match_date`, `home/away_team(_id)`, `home/away_score`, `tournament_name`, `season_id/name` |
| Jogador | `player_id`, `player_name`, `side`, `team_name`, `shirt_number`, `position`, `is_substitute` |
| Volume | `minutes_played`, `touches`, `rating` (SofaScore) |
| Finalização | `goals`, `assists`, `shots_total`, `shots_on_target`, `shots_off_target`, `expected_goals`, `expected_assists` |
| Passe | `passes_total`, `passes_accurate`, `pass_accuracy_pct`, `key_passes`, `big_chance_created/missed` |
| Duelo/defesa | `duels_won/lost`, `aerial_won/lost`, `tackles_total/won`, `interceptions`, `clearances`, `blocks`, `blocked_shots`, `ball_recovery` |
| Disciplina | `yellow_cards`, `red_cards`, `card_minutes` (minutos dos cartões) |
| Outros | `fouls_committed`, `fouls_suffered`, `offsides`, `possession_lost`, `stats_raw_json` (dump bruto p/ auditoria) |

Os CSVs usam `utf-8-sig`, abrindo corretamente no Excel/Sheets com acentos.
Para análise em pandas basta `pd.read_csv("data/...csv")`.
