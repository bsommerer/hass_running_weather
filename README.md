# hass_running_weather

Custom Home Assistant sensor, der stündliche Wettervorhersagen relativ bewertet und ein "Laufwetter"-Widget erzeugt. Das Widget zeigt, ob es gerade gutes Laufwetter ist, und listet die besten Zeitfenster für heute und die kommenden Tage auf.

## Idee: Was ist "gutes" Laufwetter?
Statt fixer Schwellwerte vergleicht der Sensor alle Vorhersage-Slots der nächsten Stunden/Tage miteinander und vergibt Scores (0–100). Faktoren und Gewichtung:

* **Temperatur** – je näher am Mittelwert der Vorhersageperiode, desto besser (Standard: 30 %). Bei Bedarf kann stattdessen `low_is_better` oder `high_is_better` genutzt werden.
* **Luftfeuchtigkeit** – niedrigere Feuchte ist angenehmer (Standard: 20 %). Default-Funktion: `low_is_better`.
* **Wind** – weniger Wind ist besser (Standard: 15 %). Default-Funktion: `low_is_better`.
* **Niederschlag/Regelwahrscheinlichkeit** – trockene Slots werden bevorzugt (Standard: 15 %). Default-Funktion: `low_is_better`.
* **Sonne** – bevorzugt sonnigere Slots anhand Cloud Coverage/UV (Standard: 10 %, `high_is_better`).
* **Bodenverhältnisse** – feuchte oder schneebedeckte Böden werden abgestraft (Standard: 10 %, `low_is_better`).
* **Tageslicht-Bonus** – leichte Bevorzugung für Zeiten zwischen 06–21 Uhr (Standard: 5 % Zusatzbonus, deaktivierbar über `daylight_bonus: 0`).

Die Prozentwerte werden intern normalisiert, sodass du die Priorisierung in Prozent angeben kannst, ohne exakt auf 100 % zu kommen. Über `scoring` lässt sich pro Faktor steuern, ob eher ein Mittelwert (`relative_center`), niedrige (`low_is_better`) oder höhere Werte (`high_is_better`) bevorzugt werden.
* **Wetterzustände** – Regen, Schnee oder Gewitter erhalten relative Strafpunkte.

Durch diese relative Betrachtung werden automatisch die bestmöglichen Slots im aktuellen Wetterumfeld hervorgehoben, auch wenn die Bedingungen insgesamt heiß/kühl oder windig sind.

## Installation
### Über HACS (empfohlen)
1. Füge in HACS unter **Integrations → Custom repositories** das Repository `https://github.com/example/hass_running_weather` mit Typ **Integration** hinzu.
2. Suche in HACS nach **Running Weather** und installiere die Integration.
3. Starte Home Assistant neu. Aktualisierungen kannst du anschließend direkt über HACS einspielen.

### Manuell
1. Kopiere den Ordner `custom_components/running_weather` in deinen `config/custom_components` Ordner von Home Assistant.
2. Starte Home Assistant neu.

## Konfiguration (YAML)
Füge in deine `configuration.yaml` hinzu und passe die Weather-Entity an:

```yaml
sensor:
  - platform: running_weather
    name: Laufwetter
    weather_entity: weather.home
    hourly_windows: 120  # Anzahl Stunden Vorhersage, die relativ bewertet werden (12–240)
    weights:              # optionale Gewichtung der Faktoren in Prozent
      temperature: 30
      humidity: 20
      wind_speed: 15
      precipitation: 15
      sunshine: 10
      ground: 10
    daylight_bonus: 5    # Zusatzbonus in Prozentpunkten tagsüber (0-30)
    scoring:              # optionale Bewertungsfunktionen je Faktor
      temperature: relative_center  # andere Optionen: low_is_better, high_is_better
      humidity: low_is_better
      wind_speed: low_is_better
      precipitation: low_is_better
      sunshine: high_is_better      # nutzt cloud_coverage oder uv_index
      ground: low_is_better         # leitet Bodenfeuchte aus Niederschlag + Wetterzustand ab
```

Der Sensor erzeugt die Entity `sensor.laufwetter` (oder den von dir gewählten Namen) mit:

* **State**: `sehr gut`, `ok` oder `schlecht` im Vergleich zu den besten Slots der Periode.
* **Attribute**: aktueller Score, aktuelle Bedingungen, `best_today` und `best_week` mit Zeitfenstern inkl. Begründung und Basisdaten.

## Lovelace-Widget Beispiel
Die Datei [`lovelace/running_weather_example.yaml`](lovelace/running_weather_example.yaml) enthält ein fertiges Card-Setup:

```yaml
type: vertical-stack
cards:
  - type: entity
    entity: sensor.running_weather
    name: Laufwetter jetzt
    icon: mdi:run
  - type: markdown
    title: Beste Slots heute
    content: >
      {% set slots = state_attr('sensor.running_weather', 'best_today') or [] %}
      {% if not slots %}
      Keine Daten verfügbar. Stelle sicher, dass dein Wetterdienst Vorhersagedaten liefert.
      {% else %}
      {% for slot in slots %}
      • **{{ as_timestamp(slot.time) | timestamp_custom('%H:%M') }}** – Score {{ slot.score }} | {{ slot.reason }}
      {% endfor %}
      {% endif %}
  - type: markdown
    title: Top-Slots in den nächsten 7 Tagen
    content: >
      {% set slots = state_attr('sensor.running_weather', 'best_week') or [] %}
      {% if not slots %}
      Keine Daten verfügbar.
      {% else %}
      {% for slot in slots %}
      • **{{ as_timestamp(slot.time) | timestamp_custom('%a %H:%M') }}** – Score {{ slot.score }} | {{ slot.reason }}
      {% endfor %}
      {% endif %}
```

Ein Tap auf die Karte zeigt sofort, ob du jetzt laufen solltest, und darunter siehst du die nächsten idealen Zeitfenster.

## Hinweise zur Datenbasis
* Der Sensor benötigt eine Wetter-Entity mit stündlicher Vorhersage (`forecast`).
* Die Berechnung ist robust gegen fehlende Einzelwerte, nutzt aber nur vorhandene Felder (Temperatur, Luftfeuchte, Wind, Niederschlag/Probability, Condition).
* Sonnenanteil nutzt standardmäßige Felder (`cloud_coverage` oder `uv_index`). Bodenverhältnisse werden rein aus Vorhersagedaten (Niederschlag, Condition) abgeschätzt – es werden keine zusätzlichen Sensoren benötigt.
* Aktualisierung alle 30 Minuten; `hourly_windows` steuert den betrachteten Zeitraum.

## Warum relativ?
Absolute Schwellwerte (z.B. immer 15 °C) sind oft unpassend. Durch die relative Bewertung innerhalb der nächsten Stunden/Woche wird das **beste verfügbare** Wetter hervorgehoben – egal ob Hochsommer oder Winter. So bekommst du immer den optimalen Slot aus dem aktuellen Wetterumfeld an deinem Standort.
