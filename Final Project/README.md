# Bomberman RL – World-Model-Agent

Dieses Projekt erweitert das bereitgestellte Bomberman-Framework um einen
CPU-tauglichen Reinforcement-Learning-Agenten. Der aktuelle Agent kombiniert
ein kleines Double-DQN mit einem symbolischen World Model. Ein visueller
Encoder ist nicht nötig, weil der vollständige Spielzustand bereits als
strukturierte Daten vorliegt.

## Aktueller Stand

Der verwendete Agent befindet sich in `agent_code/world_model_agent`. Trainierte
Checkpoints werden bewusst nicht mit Git versioniert. Lokal verwendet der Agent
standardmäßig:

```text
agent_code/world_model_agent/world-model.pt
```

Diese Datei muss lokal durch Training erzeugt oder aus einem gesicherten
Checkpoint bereitgestellt werden. Der Push enthält ausschließlich Quellcode,
Tests und Dokumentation.

Die wichtigsten Bestandteile sind:

- neuronale Q-Funktion `42 -> 64 -> 64 -> 6` auf Basis von scikit-learn,
- Double-DQN mit Target-Netzwerk und 5-Step-Returns,
- Replay Buffer mit dauerhaft gespeicherten seltenen positiven und negativen
  Erfahrungen,
- vollständiges symbolisches World Model für Spielzustand, Reward,
  Terminierung und gültige Aktionen,
- separate Köpfe für Todesrisiko, Suizidrisiko und verzögerte
  Bomben-Outcomes,
- separates Gegner-Modell für Bewegung, Bombenplatzierung und Ausscheiden,
- ein modellgenerierter Dyna-Übergang pro realem Übergang.

Alle derzeit vorhandenen 46 Unit-Tests laufen erfolgreich.

## Policy-Features

Die Policy verwendet einen kompakten Zustand mit 42 Features:

| Indizes | Bedeutung |
|---|---|
| 0–3 | Richtung zur nächsten Münze |
| 4–7 | Inhalt der vier Nachbarfelder |
| 8–11 | Richtung des vorherigen Feldes |
| 12 | begrenzte Distanz zur nächsten Münze |
| 13 | aktuelle Gefahrenstufe |
| 14–18 | sichere Bewegungs- und Warteaktionen |
| 19–23 | empfohlene Fluchtrichtung |
| 24 | Bombe verfügbar |
| 25 | Bombe würde Kiste oder Gegner treffen |
| 26–29 | Richtung zum nächsten Gegner |
| 30 | begrenzte Gegnerdistanz |
| 31–34 | Gegner auf direkt benachbarten Feldern |
| 35–38 | Gegner in den vier Bombenstrahlen |
| 39 | freie Fluchtrouten des nächsten Gegners |
| 40 | Gegner ist geometrisch eingeschlossen |
| 41 | eigene Fluchtrouten nach einer Bombe |

Gefährliche Aktionen werden nicht hart aus der Aktionsmenge entfernt. Der
Agent darf sie wählen und soll ihre Folgen aus Rewards und Risikomodellen
lernen.

## World Model

Das World Model arbeitet zusätzlich zum kompakten Policy-Zustand mit dem
vollständigen symbolischen Spielfeld. Es kodiert sieben Board-Kanäle:

- Feldaufbau,
- Münzen,
- Bomben und Timer,
- Explosionen,
- eigene Position,
- Gegnerpositionen,
- Bombenverfügbarkeit der Gegner.

Die verzögerten Bomben-Outcomes sagen vier Größen voraus:

- eigene Überlebenswahrscheinlichkeit,
- Wahrscheinlichkeit einer zerstörten Kiste,
- Wahrscheinlichkeit eines Gegner-Kills,
- Wahrscheinlichkeit, dass ein bedrohter Gegner entkommt.

Seltene Kill- und Failure-Beispiele werden in zusätzlichen persistenten
Puffern behalten, damit sie beim Überschreiben des normalen FIFO-Replays nicht
verloren gehen. Das Gegner-Modell wird faktorisiert aus der Sicht jedes Gegners
trainiert.

Die letzte Evaluation des empfohlenen Checkpoints ergab:

| Horizont | Policy-Feature-Genauigkeit | vollständiger Zustand exakt |
|---:|---:|---:|
| 1 | 94,5 % | 19,0 % |
| 2 | 92,0 % | 9,4 % |
| 3 | 91,1 % | 9,6 % |

Kurze Vorhersagen sind damit nützlich. Für lange offene Rollouts ist das Modell
noch nicht genau genug, weshalb der Standardhorizont aktuell bei einem Schritt
liegt.

## Vergleich gegen Rule-Based-Agents

Die folgenden Werte sind Mittelwerte aus drei festen Seeds mit jeweils 50
Runden gegen drei `rule_based_agent`-Gegner:

| Variante | Score | Coins | Kills | Suizide |
|---|---:|---:|---:|---:|
| World-Model-Agent | 40,33 | 17,00 | 4,67 | 7,67 |
| derselbe Checkpoint, nur Q-Netz | 33,33 | 18,33 | 3,00 | 13,33 |
| Rule-Based-Agent, Mittel pro Gegner | 185,33 | 139,78 | 9,11 | 24,00 |

Das World Model verbessert den Score um etwa 21 %, erhöht die Kills und senkt
die Suizide um etwa 42 % gegenüber der reinen Q-Policy. Der Agent ist dennoch
noch nicht konkurrenzfähig: Er wartet zu häufig und sammelt deutlich zu wenige
Münzen.

Die aktuell besten Inferenzparameter sind:

```text
DYNA_BOMB_KILL_PROBABILITY_THRESHOLD=0.10
DYNA_BOMB_KILL_OUTCOME_VALUE=25.0
DYNA_DEATH_RISK_PENALTY=18.0
```

## Erkenntnisse aus zusätzlicher Datensammlung

Mehr Daten sind hilfreich, aber die Datenverteilung ist entscheidend:

- 50 Runden gegen Peaceful-Agents erhöhten die Coin-Zahl leicht,
  verschlechterten aber Kampfleistung und Sicherheit gegen Rule-Based-Agents.
- Ein anschließendes gemischtes Curriculum konnte die ursprüngliche Leistung
  nicht wiederherstellen. Gleichzeitiges weiteres Q-Training führte zu
  Catastrophic Forgetting.
- Mit eingefrorenem Q-Netz konnte das World Model separat weitertrainiert
  werden. Die vollständige Gegnerprognose stieg dabei von ungefähr 46 % auf
  51 %, während Bomben- und Kill-Prognosen nicht besser wurden.
- Im aktuellen Bomben-Replay sind nur ungefähr 40 echte Kills unter 5.000
  Outcomes. Weitere Daten sollten deshalb gezielt seltene Kampf-, Flucht- und
  Suizidsituationen erzeugen statt hauptsächlich normale Bewegungen und
  `WAIT`-Übergänge zu sammeln.

Der experimentelle V19-Curriculum-Checkpoint wurde daher nicht als neuer
Standard übernommen. Der V18/V16-kompatible Checkpoint bleibt der beste
validierte Stand.

## Wichtige Befehle

Evaluation gegen drei Rule-Based-Agents:

```bash
python main.py play \
  --agents world_model_agent rule_based_agent rule_based_agent rule_based_agent \
  --seed 396 --n-rounds 50 --no-gui \
  --save-stats results/world-model-eval.json
```

Normales Training mit einem Dyna-Planungsschritt:

```bash
DYNA_PLANNING_ROLLOUTS=1 python main.py play \
  --agents world_model_agent rule_based_agent peaceful_agent peaceful_agent \
  --train 1 --seed 896 --n-rounds 50 --no-gui \
  --save-stats results/world-model-train.json
```

Nur World-Model-Daten sammeln, ohne die Q-Policy zu verändern:

```bash
DYNA_FREEZE_Q_NETWORK=1 python main.py play \
  --agents world_model_agent rule_based_agent rule_based_agent rule_based_agent \
  --train 1 --seed 996 --n-rounds 50 --no-gui \
  --save-stats results/world-model-data.json
```

World Model auswerten:

```bash
python -m agent_code.world_model_agent.evaluate_world_model \
  agent_code/world_model_agent/world-model.pt \
  --max-horizon 3 --max-starts 500 --seed 48 \
  --json-output results/world-model-evaluation.json
```

Tests ausführen:

```bash
python -m unittest tests.test_world_model_agent -v
```

## Nächste sinnvolle Schritte

1. Q-Policy und World-Model-Datensammlung weiterhin getrennt behandeln.
2. Seltene Kill-, Trap-, Flucht- und Suizidübergänge gezielt sammeln und in
   getrennten Puffern behalten.
3. Risiko- und Kill-Wahrscheinlichkeiten auf einem zeitlich getrennten
   Validierungsdatensatz kalibrieren.
4. Änderungen immer auf denselben Seeds mit einer reinen Q-Ablation
   vergleichen.
5. Erst längere Dyna-Rollouts oder eine GRU testen, wenn vollständige Bomben-,
   Positions- und Gegnerobjekte über mehrere Schritte zuverlässig vorhergesagt
   werden.

Ausführliche Ergebnisdateien befinden sich im Ordner `results/`.
