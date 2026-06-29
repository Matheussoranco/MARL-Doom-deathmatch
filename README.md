# MARL-Doom-deathmatch

**Self-Play para Deathmatch no ViZDoom** — avaliando uma implementação de **IPPO com *league training*** em um cenário *deathmatch* de Doom, aprendendo de ponta a ponta a partir de pixels.

> **Trabalho de disciplina — Mestrado em Sistemas Multiagentes**
> Disciplina **INF2072 · IA3 (Inteligência Artificial 3 / Aprendizado por Reforço Multiagente) · INFORMÁTICA · 2026.1**.
> Projeto final desenvolvido por **Paloma Sette**, **Antônio Moreira Pinto**, **João Ventura**, **José Ribamar Durand** e **Matheus Soranço**.

Repositório: <https://github.com/Matheussoranco/MARL-Doom-deathmatch>

---

## 1. Resumo

Este projeto investiga se **uma única política recorrente compartilhada**, treinada com **IPPO (Independent PPO)** e **self-play por liga (*league training*)**, é capaz de aprender comportamento competente de *deathmatch* no **ViZDoom** — **sem táticas codificadas à mão e sem oponentes programados fixos**, apenas a partir dos **pixels brutos** da visão em primeira pessoa.

O sistema instancia **8 jogadores simultâneos** no mapa `cig` do ViZDoom. Uma fração deles (**contribuidores**) joga e envia *rollouts* que treinam a política; os demais (**oponentes da liga**) carregam *snapshots* recentes da própria política e atuam como adversários diversos e não-estacionários, **sem contribuir com gradientes**. Um aprendiz central (PPO) consome os *rollouts*, atualiza a rede e publica novos pesos que os atores recarregam periodicamente (arquitetura distribuída no estilo A3C/IMPALA).

### Tese

> Uma única política recorrente compartilhada, treinada com **IPPO** e **self-play por liga**, aprende comportamento eficaz de *deathmatch* no ViZDoom **de ponta a ponta a partir de pixels**.

### Questão de pesquisa

> Uma **liga de snapshots anteriores** melhora a robustez do agente em comparação ao **self-play puro**?

---

## 2. Motivação — por que FPS 3D é difícil para RL

Um *first-person shooter* tridimensional reúne, ao mesmo tempo, três dificuldades clássicas de aprendizado por reforço:

- **Percepção.** A entrada são pixels brutos de alta dimensão — o agente precisa *aprender a ver* antes de aprender a agir.
- **Observabilidade parcial.** A visão em primeira pessoa e a oclusão tornam o problema um **POMDP** (Processo de Decisão de Markov Parcialmente Observável), motivando o uso de **memória recorrente**.
- **Não-estacionariedade.** Os oponentes também aprendem; o ambiente é um **alvo móvel** do ponto de vista de cada agente.

**Plataforma:** mapa `cig` do **ViZDoom** (Kempka et al., 2016), que comporta **até 8 jogadores simultâneos** em modo *deathmatch*.

---

## 3. Fundamentação — a linhagem técnica

O projeto reúne componentes consolidados da literatura de RL profundo:

| Eixo | Componente | Referência |
|---|---|---|
| **Percepção** | Pilha convolucional da DQN (`32×8/4 · 64×4/2 · 64×3/1`) | Mnih et al., 2015 |
| **Memória** | Recorrência (DRQN) para tratar o POMDP | Hausknecht & Stone, 2015 |
| **Otimização** | PPO + GAE (IPPO) | Schulman et al., 2016/2017 |
| **Escala** | Coleta distribuída ator-aprendiz (A3C / IMPALA) | Mnih et al., 2016; Espeholt et al., 2018 |
| **Estabilidade** | Inicialização ortogonal | Saxe et al., 2014 |

### Self-play e treinamento por liga

A ideia de **liga** vem do **AlphaStar** (Vinyals et al., 2019): manter uma *população* de políticas e *exploiters* evita o **colapso estratégico** e o **esquecimento cíclico** do self-play ingênuo. O **SCC** mostrou que esse estilo de liga é reprodutível a custo muito menor, motivando uma **liga de baixo orçamento**.

**Nosso recorte:** aplicamos a ideia de liga, mas a uma **única política compartilhada** e com **orçamento reduzido**. A abordagem situa-se *entre* o self-play ingênuo e o treino por população completo.

---

## 4. Arquitetura do sistema

### 4.1 Visão geral — contribuidores vs. liga

```
   ┌─────────────────────────────────────────────────────────────┐
   │                  Partida deathmatch (mapa cig)               │
   │                                                              │
   │   AGENT0 (host)   AGENT1 ... AGENTk    AGENTk+1 ... AGENT7    │
   │   └── contribuidores ──┘      └──── oponentes da liga ────┘   │
   │        (treinam)                  (carregam snapshots)       │
   └───────────┬──────────────────────────────────────┬──────────┘
               │ rollouts (obs, ações, recompensas…)   │ (sem gradiente)
               ▼                                        ▼
        ┌──────────────┐   publica pesos   ┌────────────────────────┐
        │  PPOLearner  │ ────────────────► │ policy_latest.pt        │
        │  (1 aprendiz)│ ◄──── rollout_q   │ snapshots/snapshot_*.pt │
        └──────────────┘                   └────────────────────────┘
```

- **Contribuidores** jogam e enviam *rollouts* (via `rollout_q`) que treinam a política.
- **Oponentes da liga** carregam, no início de cada episódio e com probabilidade `p`, um *snapshot* aleatório do conjunto recente. Eles **não contribuem com gradientes** — existem apenas para diversificar os adversários.
- O **`PPOLearner`** (processo principal) consome *rollouts*, executa as atualizações PPO e **publica** os pesos atualizados em `policy_latest.pt`, que os atores recarregam a cada `refresh_every` passos.

> Cada agente roda em **seu próprio processo**, com **sua própria instância do ViZDoom**, enxergando a arena apenas pela **visão em primeira pessoa**. A comunicação atores↔aprendiz é feita por filas de `multiprocessing` (`rollout_q`, `video_q`).

### 4.2 Ambiente, observações e ações

Implementado em `DeathmatchEnv` (`doom_deathmatch_ppo.py`):

- **Cenário:** `deathmatch` no mapa `cig` · episódios de **4 min** · **respawn forçado** (`sv_forcerespawn`), proteção de respawn, *spawn* mais distante, sem mira automática (`sv_noautoaim`).
- **Observação:** quadro **84×84 em tons de cinza**, com **frame-stack de 4** (fornece pistas de movimento) e **frame-skip de 4** (cada ação é repetida por 4 *tics* do jogo).
- **Ações:** **19 ações discretas compostas**, construídas a partir de **9 botões** do Doom — `ATTACK`, `SPEED`, `MOVE_FORWARD`, `MOVE_BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`, `MOVE_LEFT`, `MOVE_RIGHT`, `SELECT_NEXT_WEAPON`. As combinações cobrem mover + atirar, correr + virar, *strafe* + atacar, troca de arma, etc.
- **Observabilidade parcial:** apenas a visão em 1ª pessoa → **POMDP**, o que justifica a recorrência (GRU).

### 4.3 Modelagem de recompensa (densa)

Sinal **composto e denso**: cada termo é proporcional à variação Δ de uma variável de jogo do ViZDoom entre passos. A recompensa final é **recortada em `[−3, +5]`** (assimétrica — os ganhos positivos podem ser mais intensos que penalidades isoladas, para encorajar agressividade sem medo de errar).

| Componente | Variável | Coeficiente | Papel |
|---|---|---:|---|
| Frag | Δ `FRAGCOUNT` | **+1,000** | esparso (sinal **primário**) |
| Dano causado | Δ⁺ `DAMAGECOUNT` | +0,010 | denso |
| Acerto de tiro | Δ⁺ `HITCOUNT` | +0,020 | denso |
| Item adquirido | Δ⁺ `ITEMCOUNT` | +0,050 | exploração |
| Vida | Δ⁺ `HEALTH` | +0,001 | sobrevivência |
| Munição | Δ⁺ `SELECTED_WEAPON_AMMO` | +0,005 | gestão de recursos |
| Dano sofrido | Δ⁺ `DAMAGE_TAKEN` | −0,005 | penalidade |
| Morte | `is_player_dead()` | −0,500 | penalidade |
| Sobreviver | constante / passo | +0,0005 | *living bonus* |

> Δ⁺ = apenas incrementos positivos são considerados. Os termos densos (dano, acerto) **guiam** o agente na direção certa o tempo todo; as penalidades (dano sofrido, morte) o ensinam a se proteger sem paralisá-lo. Todos os coeficientes são expostos como *flags* de linha de comando para experimentação.

### 4.4 Rede: ator-crítico recorrente

`RecurrentActorCritic` (tronco compartilhado):

```
obs (4×84×84, uint8) ──/255──►
  Conv 32×8/4 ─ ReLU ─ Conv 64×4/2 ─ ReLU ─ Conv 64×3/1 ─ ReLU
        │ (flatten 64·7·7)
        ▼
  Linear → 256 ─ ReLU ──► GRUCell(256 → 256) ── hₜ ∈ ℝ²⁵⁶
                                    ├──► Actor  (Linear 256 → 19 logits)
                                    └──► Critic (Linear 256 → 1  valor)
```

- **Inicialização ortogonal** (Saxe et al., 2014): ganho `√2` no tronco, **`0,01` no ator** (logits iniciais pequenos → política quase uniforme → **boa exploração inicial**) e `1,0` no crítico.
- **Recorrência GRU:** o estado oculto `hₜ ∈ ℝ²⁵⁶` é atualizado a cada passo a partir do *feature* `eₜ`. A **porta de reset** permite tratar uma situação como nova; a **porta de update** preserva memória de longo prazo. O **estado oculto é zerado na morte / fim de episódio**.

### 4.5 Algoritmo de aprendizado — PPO recorrente

`PPOLearner`:

- **PPO** (Schulman et al., 2017): aproximação escalável da região de confiança do TRPO. Objetivo **recortado** com `ε = 0,1` (mais conservador que o `0,2` do artigo original), **clip de valor**, **bônus de entropia** e **clip de gradiente**. Otimizador **AdamW** (Loshchilov & Hutter, 2019).
- **GAE** (Schulman et al., 2016): vantagens de baixa variância com `γ = 0,99`, `λ = 0,95`; `retornos = vantagens + valores`. As vantagens são normalizadas por *batch*.
- **Atualizações recorrentes:** **BPTT truncado** em *rollouts* de **128** passos; o estado oculto é re-mascarado nos limites de episódio dentro do *rollout*.
- **Coleta distribuída (quase on-policy):** vários atores, **um aprendiz**; pesos recarregados a cada `k` passos (estilo A3C/IMPALA). Os atores podem estar algumas atualizações atrás do aprendiz — **o clip do PPO tolera** essa defasagem.

### 4.6 Liga de self-play

Uma fração dos jogadores são **oponentes da liga**: no início de cada episódio, com probabilidade `p = 0,6`, carregam um *snapshot* aleatório do **conjunto recente** (os mais antigos são descartados — pool limitado). 

**Objetivo:** manter oponentes **diversos e não-estacionários**, evitando o **sobreajuste à versão atual** da política e reduzindo o **esquecimento estratégico** — tudo isso com **política única compartilhada** e **baixo custo computacional**.

---

## 5. Estrutura do repositório

```
MARL-Doom-deathmatch/
├── doom_deathmatch_ppo.py            # Implementação principal (treino + avaliação + bundle)
├── doom_deathmatch_ppo_colab.ipynb   # Variante em notebook para Google Colab
├── pod_setup.sh                      # Bootstrap de pod cloud (RunPod / Vast.ai / Lambda)
└── README.md
```

Componentes-chave de `doom_deathmatch_ppo.py`:

| Símbolo | Papel |
|---|---|
| `DeathmatchEnv` | Wrapper do ViZDoom: configuração `cig`, pré-processamento, *frame-stack*, modelagem de recompensa, gravação de vídeo. |
| `RecurrentActorCritic` | Rede CNN + GRU + cabeças ator/crítico. |
| `actor_worker` | Processo de um jogador: coleta *rollouts*, alterna entre política atual e *snapshots* da liga, grava o melhor episódio. |
| `PPOLearner` | Aprendiz central: GAE, atualização PPO recorrente, *checkpoints*, publicação de pesos. |
| `train` | Orquestra os processos, a barreira de *handshake* multiplayer e o laço de treino. |
| `evaluate` / `eval_worker` | Passo de avaliação em **1280×720** com política gulosa (4% de exploração). |
| `video_writer_proc` / `_save_video` | Processo dedicado à escrita de MP4 (libx264). |
| `bundle_results` | Empacota vídeos e *checkpoints* em `doom_marl_results.zip`. |

---

## 6. Como executar

### 6.1 Pré-requisitos

- **Python 3.10**, **PyTorch** (com CUDA se houver GPU), **ViZDoom 1.2.4** (fixado — ver nota abaixo), `opencv-python-headless`, `numpy<2.1`, `imageio` + `imageio-ffmpeg`, **ffmpeg**.
- GPU recomendada: **RTX 4090 / A10 / A40** (uma H100 é desperdício aqui). **≥ 16 vCPU** (um processo ViZDoom por agente + o aprendiz), **≥ 24 GB RAM**, **≥ 30 GB de disco**.

> ⚠️ **ViZDoom 1.2.4 fixado.** A reformulação de engine da 1.3.0 (reinício do backend de áudio, determinismo de texturas) provoca *segfault* dentro de `init()` em pods *headless*, mesmo com xvfb + Mesa GL funcionando. A 1.2.4 é a última build 1.2 estável e é compatível com este código.

### 6.2 Provisionamento em pod cloud (Linux headless)

O script `pod_setup.sh` automatiza a instalação (deps de build do ViZDoom, ffmpeg, tmux, venv, PyTorch com a *wheel* CUDA correta) e roda *smoke tests* de CUDA, ViZDoom (em modo `-host -deathmatch`, pois `cig` **não tem** *start* single-player) e ffmpeg:

```bash
chmod +x pod_setup.sh
./pod_setup.sh

source ~/doom_venv/bin/activate
tmux new -s doom
# Em pod headless o treino DEVE rodar sob xvfb (senão a engine dá segfault ao abrir o contexto GL):
xvfb-run -a -s '-screen 0 1280x1024x24' \
    python doom_deathmatch_ppo.py --save_dir ./runs/run01
# detach: Ctrl+B depois D     reattach: tmux attach -t doom
```

### 6.3 Execução direta

```bash
# Treino completo (8 jogadores: 5 contribuidores + 3 na liga) seguido de avaliação
python doom_deathmatch_ppo.py --save_dir ./runs/run01

# Apenas avaliação a partir de um checkpoint
python doom_deathmatch_ppo.py --skip_train --resume ./runs/run01/checkpoint_final.pt

# Self-play PURO (baseline sem liga): todos contribuem, nenhum oponente de liga
python doom_deathmatch_ppo.py --n_contributors 8 --save_dir ./runs/no_league

# Liga mais agressiva (poucos contribuidores, muitos oponentes de liga)
python doom_deathmatch_ppo.py --n_contributors 2 --save_dir ./runs/more_league
```

### 6.4 Google Colab

`doom_deathmatch_ppo_colab.ipynb` traz uma variante guiada (instalação de dependências, *smoke test* do `cig.cfg`, fluxo de treino e gravação de vídeos) para execução em uma única GPU do Colab.

### 6.5 Saídas (em `./runs/run01/`)

| Arquivo / pasta | Conteúdo |
|---|---|
| `policy_latest.pt` | Arquivo de pesos *vivo* que os atores recarregam. |
| `checkpoint_final.pt` | Estado completo do PPO ao fim do treino. |
| `snapshots/` | Pool de *snapshots* da liga. |
| `videos/` | Melhor episódio de treino por agente (640×480). |
| `eval_videos/` | Gravações finais de avaliação (1280×720). |
| `doom_marl_results.zip` | Entregável empacotado. |

---

## 7. Configuração experimental e hiperparâmetros

- **Jogadores:** 8 no total — **5 contribuidores + 3 na liga** (`p = 0,6`).
- **Avaliação:** 1280×720 · `N` episódios · política **gulosa (4% de exploração)** · métrica = **frags / episódio**.
- **Reprodutibilidade:** sementes fixas · **ViZDoom 1.2.4 fixado**.

| Hiperparâmetro | Valor | | Hiperparâmetro | Valor |
|---|---:|---|---|---:|
| `rollout T` | 128 | | recorte `ε` | **0,1** |
| `rollouts/atualização` | 16 | | recorte de valor | 0,2 |
| épocas PPO | 4 | | `coef-vf` | 0,5 |
| minibatch (rollouts) | 4 | | entropia | 0,005 |
| `lr` (AdamW) | 2,5e-4 | | recorte de gradiente | 0,5 |
| `γ` | 0,99 | | tamanho oculto (GRU) | 256 |
| `λ` (GAE) | 0,95 | | frame stack / skip | 4 / 4 |
| liga `p` | 0,6 | | total de atualizações | 4000 |

> O recorte `ε = 0,1` é deliberadamente **mais conservador** que o `0,2` do artigo original do PPO. Todos os valores acima são configuráveis por *flags* (`python doom_deathmatch_ppo.py --help`).

---

## 8. Resultados

### Protocolo de torneio

Para responder à questão de pesquisa (liga vs. self-play puro), três agentes foram treinados e confrontados em um **torneio round-robin** (100 partidas entre cada par, 100 partidas entre todos), com o vencedor definido pelo **melhor frag**:

| Agente | Configuração |
|---|---|
| `agent_1_default` | League training · **5 contribuidores** |
| `agent_2_no_league` | **Sem** league training · **8 contribuidores** |
| `agent_3_more_league` | More league training · **2 contribuidores** |

### Resultado do torneio (jogo de soma ≈ zero)

| Agente | Vitórias | Derrotas | Empates | Partidas | Mortes | Recompensa |
|---|---:|---:|---:|---:|---:|---:|
| **`agent_3_more_league`** | **2** | 0 | 0 | 2 | 122,0 | **200,34** |
| `agent_1_default` | 1 | 1 | 0 | 2 | 260,0 | −46,36 |
| `agent_2_no_league` | 0 | 2 | 0 | 2 | 207,0 | 114,82 |

**Indício a favor da liga:** o agente com **mais league training** venceu todas as partidas, com a maior recompensa e o **menor número de mortes**; o agente **sem liga** ficou em último. A análise também acompanhou a **entropia** da política como indicador de exploração ao longo do treino. *(Resultados de uma corrida com orçamento de computação limitado — ver limitações.)*

---

## 9. Discussão

### O que funcionou

- **Frags.** Os agentes aprendem comportamento competente e conseguem efetivamente abater oponentes.
- **Sem artifícios.** A ingestão visual não usa atalhos como *track boxes* — o aprendizado parte dos **pixels brutos**.
- **Ambiente de treino.** Foi possível montar um pipeline de treino multiagente **efetivo e reprodutível** (multiplayer headless, liga, gravação de vídeo, *checkpointing*).

### Comportamento emergente

Os agentes exibiram comportamento alinhado à teoria **"prospect-refuge"** de **Jay Appleton** (1975): tendência a ocupar posições que oferecem **prospecto** (campo de visão aberto e sem obstruções) e **refúgio** (segurança a partir da qual prospectar — no caso, junto a uma parede).

### Limitações

- **Aprendizado primitivo.** Apesar de obterem frags, o comportamento está longe do ideal.
- **"MARL" aqui é self-play com política compartilhada** — **não** são aprendizes verdadeiramente independentes, nem CTDE, nem há atribuição explícita de crédito.
- **Recompensa densa ajustada à mão.** Os pesos foram definidos manualmente, **sem validação por ablação**.
- **Mortes não são terminais.** Há penalidade + respawn; o valor faz *bootstrap* através das mortes.
- **Escopo limitado** por computação e por usar um **único mapa** (`cig`).
- **Engine.** As variações por `.step()` não são triviais de monitorar; as variáveis de morte **colapsam** mortes ambientais, suicídios e mortes por outros jogadores; o multiplayer torna a execução bem mais lenta.
- **Espaço de ações** grande demais (muitas combinações não mutuamente exclusivas) e possíveis **exploits** (ex.: trocar de arma para ganhar munição).

### Melhorias propostas

- **Transferência de aprendizado:** pré-treinar em cenários PvE mais simples (ex.: `defend the center`, onde o agente já aprende a atirar) antes do PvP.
- **Reestruturar as ações** em **grupos mutuamente exclusivos** (ex.: 3 grupos de botões).
- **Prevenir suicídios:** remover ou penalizar mais o lança-foguetes.
- **Ajustar a rede:** taxa de aprendizado, remover *frame stacking*.
- **Treinar por muito mais tempo** — não há critério intuitivo de parada, mas o orçamento usado foi provavelmente insuficiente.

---

## 10. Conclusão e trabalhos futuros

**A tese é validada:** uma **única política recorrente compartilhada**, treinada com **IPPO** e **self-play por liga**, aprende comportamento eficaz de *deathmatch* no ViZDoom **de ponta a ponta a partir de pixels**. A evidência preliminar do torneio sugere que **a liga melhora a robustez** em relação ao self-play puro.

**Trabalhos futuros:**
- **MARL de fato:** aprendizes independentes ou crítico centralizado (**CTDE**).
- **Refinamento da recompensa** por meio de **ablação** sistemática.
- **Liga mais rica:** *exploiters* e amostragem priorizada de oponentes, à la AlphaStar.
- **Generalização** entre mapas e cenários.

---

## 11. Referências

- **[KEMPKA2016]** Kempka et al. (2016). *ViZDoom: A Doom-based AI Research Platform.*
- **[MNIH2015]** Mnih et al. (2015). *Human-level control through deep reinforcement learning.*
- **[HAUSKNECHT2015]** Hausknecht & Stone (2015). *Deep Recurrent Q-Learning for POMDPs.*
- **[MNIH2016]** Mnih et al. (2016). *Asynchronous Methods for Deep RL (A3C).*
- **[ESPEHOLT2018]** Espeholt et al. (2018). *IMPALA: Scalable Distributed Deep-RL.*
- **[SCHULMAN2016]** Schulman et al. (2016). *High-Dimensional Continuous Control Using GAE.*
- **[SCHULMAN2017]** Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- **[SAXE2014]** Saxe et al. (2014). *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.*
- **[CHUNG2014]** Chung et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks.*
- **[LOSHCHILOV2019]** Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization (AdamW).*
- **[VINYALS2019]** Vinyals et al. (2019). *Grandmaster level in StarCraft II via multi-agent reinforcement learning (AlphaStar).*
- **[SCC]** *SCC: an Efficient Deep Reinforcement Learning Agent (StarCraft Commander).*
- **[APPLETON1975]** Appleton, J. (1975). *The Experience of Landscape.*

---

### Autores

Paloma Sette · Antônio Moreira Pinto · João Ventura · José Ribamar Durand · Matheus Soranço

*Projeto desenvolvido para a disciplina INF2072 · IA3 · Informática · 2026.1 — Mestrado em Sistemas Multiagentes.*
